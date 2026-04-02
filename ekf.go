package controlsys

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// EKFModel defines the nonlinear model for an Extended Kalman Filter.
type EKFModel struct {
	F    func(x, u *mat.VecDense) *mat.VecDense // state transition x_{k+1} = f(x_k, u_k)
	H    func(x *mat.VecDense) *mat.VecDense    // measurement y_k = h(x_k)
	FJac func(x, u *mat.VecDense) *mat.Dense    // Jacobian df/dx at (x, u)
	HJac func(x *mat.VecDense) *mat.Dense       // Jacobian dh/dx at x
	Q    *mat.Dense                              // process noise covariance (n×n)
	R    *mat.Dense                              // measurement noise covariance (p×p)
}

// EKF implements an Extended Kalman Filter for nonlinear systems.
type EKF struct {
	model *EKFModel
	X     *mat.VecDense // state estimate
	P     *mat.Dense    // error covariance
	n     int           // state dimension
	p     int           // measurement dimension
}

// NewEKF creates an Extended Kalman Filter from a nonlinear model,
// initial state x0, and initial covariance P0.
func NewEKF(model *EKFModel, x0 *mat.VecDense, P0 *mat.Dense) (*EKF, error) {
	if model == nil {
		return nil, fmt.Errorf("NewEKF: nil model: %w", ErrDimensionMismatch)
	}
	if model.F == nil || model.H == nil || model.FJac == nil || model.HJac == nil {
		return nil, fmt.Errorf("NewEKF: nil function in model: %w", ErrDimensionMismatch)
	}
	if x0 == nil || P0 == nil || model.Q == nil || model.R == nil {
		return nil, fmt.Errorf("NewEKF: nil matrix argument: %w", ErrDimensionMismatch)
	}

	n := x0.Len()
	if n == 0 {
		return nil, fmt.Errorf("NewEKF: zero-length state: %w", ErrDimensionMismatch)
	}

	pr, pc := P0.Dims()
	if pr != n || pc != n {
		return nil, fmt.Errorf("NewEKF: P0 is %dx%d, want %dx%d: %w", pr, pc, n, n, ErrDimensionMismatch)
	}
	qr, qc := model.Q.Dims()
	if qr != n || qc != n {
		return nil, fmt.Errorf("NewEKF: Q is %dx%d, want %dx%d: %w", qr, qc, n, n, ErrDimensionMismatch)
	}
	rr, rc := model.R.Dims()
	if rr != rc {
		return nil, fmt.Errorf("NewEKF: R is %dx%d, must be square: %w", rr, rc, ErrDimensionMismatch)
	}
	p := rr

	xCopy := mat.VecDenseCopyOf(x0)
	pCopy := mat.DenseCopyOf(P0)

	return &EKF{
		model: model,
		X:     xCopy,
		P:     pCopy,
		n:     n,
		p:     p,
	}, nil
}

// Predict performs the EKF prediction step using control input u.
func (e *EKF) Predict(u *mat.VecDense) error {
	xPred := e.model.F(e.X, u)
	A := e.model.FJac(e.X, u)

	ar, ac := A.Dims()
	if ar != e.n || ac != e.n {
		return fmt.Errorf("EKF.Predict: FJac returned %dx%d, want %dx%d: %w", ar, ac, e.n, e.n, ErrDimensionMismatch)
	}

	// P = A * P * A' + Q
	var ap mat.Dense
	ap.Mul(A, e.P)
	e.P.Mul(&ap, A.T())
	e.P.Add(e.P, e.model.Q)

	e.X = xPred
	return nil
}

// Update performs the EKF measurement update step using observation z.
func (e *EKF) Update(z *mat.VecDense) error {
	if z.Len() != e.p {
		return fmt.Errorf("EKF.Update: z length %d, want %d: %w", z.Len(), e.p, ErrDimensionMismatch)
	}

	yPred := e.model.H(e.X)
	C := e.model.HJac(e.X)

	cr, cc := C.Dims()
	if cr != e.p || cc != e.n {
		return fmt.Errorf("EKF.Update: HJac returned %dx%d, want %dx%d: %w", cr, cc, e.p, e.n, ErrDimensionMismatch)
	}

	// S = C * P * C' + R  (p×p)
	var cp mat.Dense
	cp.Mul(C, e.P)
	S := mat.NewDense(e.p, e.p, nil)
	S.Mul(&cp, C.T())
	S.Add(S, e.model.R)

	// K = P * C' * S^{-1}  (n×p)
	// From K * S = P * C', solve S' * K' = (P*C')' = C * P.
	// S is symmetric so: S * K' = C * P. Solve for K' (p×n), then transpose.
	Kt := mat.NewDense(e.p, e.n, nil)
	if err := Kt.Solve(S, &cp); err != nil {
		return fmt.Errorf("EKF.Update: innovation covariance solve failed: %w", err)
	}
	K := mat.NewDense(e.n, e.p, nil)
	K.Copy(Kt.T())

	innov := mat.NewVecDense(e.p, nil)
	innov.SubVec(z, yPred)

	var kInnov mat.VecDense
	kInnov.MulVec(K, innov)
	e.X.AddVec(e.X, &kInnov)

	// Joseph form: P = (I - K*C) * P * (I - K*C)' + K * R * K'
	IKC := mat.NewDense(e.n, e.n, nil)
	IKC.Mul(K, C)
	for i := range e.n {
		for j := range e.n {
			if i == j {
				IKC.Set(i, j, 1-IKC.At(i, j))
			} else {
				IKC.Set(i, j, -IKC.At(i, j))
			}
		}
	}

	var ikcP mat.Dense
	ikcP.Mul(IKC, e.P)
	e.P.Mul(&ikcP, IKC.T())

	var kR mat.Dense
	kR.Mul(K, e.model.R)
	var kRKt mat.Dense
	kRKt.Mul(&kR, K.T())
	e.P.Add(e.P, &kRKt)

	return nil
}

// Step performs a predict-then-update cycle.
func (e *EKF) Step(u, z *mat.VecDense) error {
	if err := e.Predict(u); err != nil {
		return err
	}
	return e.Update(z)
}
