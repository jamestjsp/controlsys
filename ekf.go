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
	Q    *mat.Dense                             // process noise covariance (n×n)
	R    *mat.Dense                             // measurement noise covariance (p×p)
}

// EKF implements an Extended Kalman Filter for nonlinear systems.
type EKF struct {
	model *EKFModel
	X     *mat.VecDense // state estimate
	P     *mat.Dense    // error covariance
	n     int           // state dimension
	p     int           // measurement dimension

	ap, cp, s     *mat.Dense
	kt, k         *mat.Dense
	ikc, ikcP, kR *mat.Dense
	kRKt          *mat.Dense
	innov, kInnov *mat.VecDense
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

	hProbe := model.H(x0)
	if hProbe.Len() != p {
		return nil, fmt.Errorf("NewEKF: H(x0) returned length %d, but R is %dx%d: %w", hProbe.Len(), p, p, ErrDimensionMismatch)
	}

	xCopy := mat.VecDenseCopyOf(x0)
	pCopy := mat.DenseCopyOf(P0)

	return &EKF{
		model:  model,
		X:      xCopy,
		P:      pCopy,
		n:      n,
		p:      p,
		ap:     mat.NewDense(n, n, nil),
		cp:     mat.NewDense(p, n, nil),
		s:      mat.NewDense(p, p, nil),
		kt:     mat.NewDense(p, n, nil),
		k:      mat.NewDense(n, p, nil),
		ikc:    mat.NewDense(n, n, nil),
		ikcP:   mat.NewDense(n, n, nil),
		kR:     mat.NewDense(n, p, nil),
		kRKt:   mat.NewDense(n, n, nil),
		innov:  mat.NewVecDense(p, nil),
		kInnov: mat.NewVecDense(n, nil),
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
	e.ap.Mul(A, e.P)
	e.P.Mul(e.ap, A.T())
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
	e.cp.Mul(C, e.P)
	e.s.Mul(e.cp, C.T())
	e.s.Add(e.s, e.model.R)

	// K = P * C' * S^{-1}  (n×p)
	// From K * S = P * C', solve S' * K' = (P*C')' = C * P.
	// S is symmetric so: S * K' = C * P. Solve for K' (p×n), then transpose.
	if err := e.kt.Solve(e.s, e.cp); err != nil {
		return fmt.Errorf("EKF.Update: innovation covariance solve failed: %w", err)
	}
	e.k.Copy(e.kt.T())

	e.innov.SubVec(z, yPred)

	e.kInnov.MulVec(e.k, e.innov)
	e.X.AddVec(e.X, e.kInnov)

	// Joseph form: P = (I - K*C) * P * (I - K*C)' + K * R * K'
	e.ikc.Mul(e.k, C)
	for i := range e.n {
		for j := range e.n {
			if i == j {
				e.ikc.Set(i, j, 1-e.ikc.At(i, j))
			} else {
				e.ikc.Set(i, j, -e.ikc.At(i, j))
			}
		}
	}

	e.ikcP.Mul(e.ikc, e.P)
	e.P.Mul(e.ikcP, e.ikc.T())

	e.kR.Mul(e.k, e.model.R)
	e.kRKt.Mul(e.kR, e.k.T())
	e.P.Add(e.P, e.kRKt)

	return nil
}

// Step performs a predict-then-update cycle.
func (e *EKF) Step(u, z *mat.VecDense) error {
	if err := e.Predict(u); err != nil {
		return err
	}
	return e.Update(z)
}
