package controlsys

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type H2SynResult struct {
	K       *System
	X       *mat.Dense
	Y       *mat.Dense
	CLPoles []complex128
}

func H2Syn(P *System, nmeas, ncont int) (*H2SynResult, error) {
	if !P.IsContinuous() {
		return nil, ErrWrongDomain
	}

	n, m, p := P.Dims()
	if n == 0 {
		return nil, ErrInvalidPartition
	}
	if ncont <= 0 || nmeas <= 0 || ncont > m || nmeas > p {
		return nil, ErrInvalidPartition
	}

	m1 := m - ncont
	m2 := ncont
	p1 := p - nmeas
	p2 := nmeas

	if m1 <= 0 || p1 <= 0 {
		return nil, ErrInvalidPartition
	}

	A := P.A
	B1 := extractBlock(P.B, 0, 0, n, m1)
	B2 := extractBlock(P.B, 0, m1, n, m2)
	C1 := extractBlock(P.C, 0, 0, p1, n)
	C2 := extractBlock(P.C, p1, 0, p2, n)
	D11 := extractBlock(P.D, 0, 0, p1, m1)
	D12 := extractBlock(P.D, 0, m1, p1, m2)
	D21 := extractBlock(P.D, p1, 0, p2, m1)
	D22 := extractBlock(P.D, p1, m1, p2, m2)

	tol := 1e-10
	d11Raw := D11.RawMatrix()
	for i := range p1 {
		for j := range m1 {
			if math.Abs(d11Raw.Data[i*d11Raw.Stride+j]) > tol {
				return nil, ErrNoFiniteH2Norm
			}
		}
	}

	stab, err := IsStabilizable(A, B2, true)
	if err != nil {
		return nil, err
	}
	if !stab {
		return nil, ErrNotStabilizable
	}

	det, err := IsDetectable(A, C2, true)
	if err != nil {
		return nil, err
	}
	if !det {
		return nil, ErrNotDetectable
	}

	// State-feedback CARE: A'X + XA - (XB2+S1)*R1^{-1}*(B2'X+S1') + Q1 = 0
	Q1 := mulDense(mat.DenseCopyOf(C1.T()), C1)
	R1 := mulDense(mat.DenseCopyOf(D12.T()), D12)
	S1 := mulDense(mat.DenseCopyOf(C1.T()), D12)

	resX, err := Care(A, B2, Q1, R1, &RiccatiOpts{S: S1})
	if err != nil {
		return nil, err
	}
	X := resX.X

	// F = -K_care (state feedback gain u = F*x)
	F := mat.NewDense(m2, n, nil)
	F.Scale(-1, resX.K)

	// Filter CARE (dual): Care(A', C2', B1*B1', D21*D21', S=B1*D21')
	Q2 := mulDense(B1, mat.DenseCopyOf(B1.T()))
	R2 := mulDense(D21, mat.DenseCopyOf(D21.T()))
	S2 := mulDense(B1, mat.DenseCopyOf(D21.T()))

	resY, err := Care(mat.DenseCopyOf(A.T()), mat.DenseCopyOf(C2.T()), Q2, R2, &RiccatiOpts{S: S2})
	if err != nil {
		return nil, err
	}
	Y := resY.X

	// L = K_dual' (observer gain, n×p2) — matches Lqe convention
	L := mat.DenseCopyOf(resY.K.T())

	// Form controller: y (p2) -> u (m2)
	d22Zero := true
	d22Raw := D22.RawMatrix()
	for i := range p2 {
		for j := range m2 {
			if math.Abs(d22Raw.Data[i*d22Raw.Stride+j]) > tol {
				d22Zero = false
				break
			}
		}
		if !d22Zero {
			break
		}
	}

	var Ak, Bk, Ck *mat.Dense
	var Dk *mat.Dense

	if d22Zero {
		// Ak = A + B2*F - L*C2
		BF := mulDense(B2, F)
		LC := mulDense(L, C2)
		Ak = mat.NewDense(n, n, nil)
		Ak.Add(A, BF)
		Ak.Sub(Ak, LC)

		Bk = denseCopy(L)
		Ck = denseCopy(F)
		Dk = mat.NewDense(m2, p2, nil)
	} else {
		DF := mulDense(D22, F)
		C2eff := mat.NewDense(p2, n, nil)
		C2eff.Add(C2, DF)
		LC := mulDense(L, C2eff)
		BF := mulDense(B2, F)
		Ak = mat.NewDense(n, n, nil)
		Ak.Add(A, BF)
		Ak.Sub(Ak, LC)

		Bk = denseCopy(L)
		Ck = denseCopy(F)
		Dk = mat.NewDense(m2, p2, nil)
	}

	K, err := New(Ak, Bk, Ck, Dk, 0)
	if err != nil {
		return nil, err
	}

	// Closed-loop A matrix for pole computation
	// CL_A = [A + B2*Dk*C2,  B2*Ck;  Bk*C2,  Ak]
	nn := 2 * n
	clA := mat.NewDense(nn, nn, nil)

	// top-left: A + B2*Dk*C2
	topLeft := denseCopy(A)
	if !d22Zero {
		BDC := mulDense(B2, mulDense(Dk, C2))
		topLeft.Add(topLeft, BDC)
	}
	setBlock(clA, 0, 0, topLeft)

	// top-right: B2*Ck
	BC := mulDense(B2, Ck)
	setBlock(clA, 0, n, BC)

	// bottom-left: Bk*C2
	BkC := mulDense(Bk, C2)
	setBlock(clA, n, 0, BkC)

	// bottom-right: Ak
	setBlock(clA, n, n, Ak)

	var eig mat.Eigen
	ok := eig.Factorize(clA, mat.EigenNone)
	if !ok {
		return nil, ErrSchurFailed
	}
	clPoles := eig.Values(nil)

	return &H2SynResult{K: K, X: X, Y: Y, CLPoles: clPoles}, nil
}

