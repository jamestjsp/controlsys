package controlsys

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/gonum/mat"
)

type HinfSynResult struct {
	K        *System
	GammaOpt float64
	X        *mat.Dense
	Y        *mat.Dense
	CLPoles  []complex128
}

func HinfSyn(P *System, nmeas, ncont int) (*HinfSynResult, error) {
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
	_ = extractBlock(P.D, p1, m1, p2, m2)

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

	gammaLB := maxSVD(D11)
	gammaUB := gammaLB*2 + 1

	for !hinfFeasible(A, B1, B2, C1, C2, D12, D21, n, m1, m2, p1, p2, gammaUB) {
		gammaUB *= 2
		if gammaUB > 1e12 {
			return nil, ErrGammaNotAchievable
		}
	}

	for gammaUB-gammaLB > 1e-6*gammaUB {
		mid := (gammaLB + gammaUB) / 2
		if hinfFeasible(A, B1, B2, C1, C2, D12, D21, n, m1, m2, p1, p2, mid) {
			gammaUB = mid
		} else {
			gammaLB = mid
		}
	}

	gamma := gammaUB
	X, Y, err := hinfSolveRiccatis(A, B1, B2, C1, C2, D12, D21, n, m1, m2, p1, p2, gamma)
	if err != nil {
		return nil, err
	}

	F := mat.NewDense(m2, n, nil)
	F.Mul(mat.DenseCopyOf(B2.T()), X)
	F.Scale(-1, F)

	L := mat.NewDense(n, p2, nil)
	L.Mul(Y, mat.DenseCopyOf(C2.T()))
	L.Scale(-1, L)

	ginv2 := 1.0 / (gamma * gamma)

	YX := mulDense(Y, X)
	nn := n
	eye := mat.NewDense(nn, nn, nil)
	for i := range nn {
		eye.Set(i, i, 1)
	}
	ZpArg := mat.NewDense(nn, nn, nil)
	ZpArg.Scale(ginv2, YX)
	ZpArg.Sub(eye, ZpArg)

	var lu mat.LU
	lu.Factorize(ZpArg)
	Zp := mat.NewDense(nn, nn, nil)
	if err := lu.SolveTo(Zp, false, eye); err != nil {
		return nil, ErrGammaNotAchievable
	}

	// Ak = A + ginv2*B1*B1'*X + B2*F + Zp*L*C2
	Ak := denseCopy(A)

	tmp1 := mulDense(B1, mat.DenseCopyOf(B1.T()))
	tmp1.Mul(tmp1, X)
	tmp1.Scale(ginv2, tmp1)
	Ak.Add(Ak, tmp1)

	Ak.Add(Ak, mulDense(B2, F))

	ZpL := mulDense(Zp, L)
	Ak.Add(Ak, mulDense(ZpL, C2))

	Bk := mulDense(Zp, L)
	Ck := denseCopy(F)
	Dk := mat.NewDense(m2, p2, nil)

	K, err := New(Ak, Bk, Ck, Dk, 0)
	if err != nil {
		return nil, err
	}

	clN := 2 * n
	clA := mat.NewDense(clN, clN, nil)
	setBlock(clA, 0, 0, A)
	BC := mulDense(B2, Ck)
	setBlock(clA, 0, n, BC)
	BkC := mulDense(Bk, C2)
	setBlock(clA, n, 0, BkC)
	setBlock(clA, n, n, Ak)

	var eig mat.Eigen
	ok := eig.Factorize(clA, mat.EigenNone)
	if !ok {
		return nil, ErrSchurFailed
	}
	clPoles := eig.Values(nil)

	return &HinfSynResult{K: K, GammaOpt: gamma, X: X, Y: Y, CLPoles: clPoles}, nil
}

func hinfFeasible(A, B1, B2, C1, C2, D12, D21 *mat.Dense, n, m1, m2, p1, p2 int, gamma float64) bool {
	_, _, err := hinfSolveRiccatis(A, B1, B2, C1, C2, D12, D21, n, m1, m2, p1, p2, gamma)
	return err == nil
}

func hinfSolveRiccatis(A, B1, B2, C1, C2, D12, D21 *mat.Dense, n, m1, m2, p1, p2 int, gamma float64) (*mat.Dense, *mat.Dense, error) {
	ginv2 := 1.0 / (gamma * gamma)

	// X-Riccati Hamiltonian:
	// H_x = [A, Gx; -C1'C1, -A']
	// Gx = ginv2*B1*B1' - B2*B2'
	B1B1t := mulDense(B1, mat.DenseCopyOf(B1.T()))
	B2B2t := mulDense(B2, mat.DenseCopyOf(B2.T()))
	Gx := mat.NewDense(n, n, nil)
	Gx.Scale(ginv2, B1B1t)
	Gx.Sub(Gx, B2B2t)

	C1tC1 := mulDense(mat.DenseCopyOf(C1.T()), C1)

	Hx := mat.NewDense(2*n, 2*n, nil)
	setBlock(Hx, 0, 0, A)
	setBlock(Hx, 0, n, Gx)
	negC1tC1 := mat.NewDense(n, n, nil)
	negC1tC1.Scale(-1, C1tC1)
	setBlock(Hx, n, 0, negC1tC1)
	negAt := mat.NewDense(n, n, nil)
	negAt.Scale(-1, mat.DenseCopyOf(A.T()))
	setBlock(Hx, n, n, negAt)

	X, err := solveHamiltonianRiccati(Hx, n)
	if err != nil {
		return nil, nil, err
	}

	// Y-Riccati Hamiltonian:
	// H_y = [A', Gy; -B1*B1', -A]
	// Gy = ginv2*C1'C1 - C2'C2
	C2tC2 := mulDense(mat.DenseCopyOf(C2.T()), C2)
	Gy := mat.NewDense(n, n, nil)
	Gy.Scale(ginv2, C1tC1)
	Gy.Sub(Gy, C2tC2)

	Hy := mat.NewDense(2*n, 2*n, nil)
	At := mat.DenseCopyOf(A.T())
	setBlock(Hy, 0, 0, At)
	setBlock(Hy, 0, n, Gy)
	negB1B1t := mat.NewDense(n, n, nil)
	negB1B1t.Scale(-1, B1B1t)
	setBlock(Hy, n, 0, negB1B1t)
	negA := mat.NewDense(n, n, nil)
	negA.Scale(-1, A)
	setBlock(Hy, n, n, negA)

	Y, err := solveHamiltonianRiccati(Hy, n)
	if err != nil {
		return nil, nil, err
	}

	// Spectral radius check: rho(X*Y) < gamma^2
	XY := mulDense(X, Y)
	var eig mat.Eigen
	ok := eig.Factorize(XY, mat.EigenNone)
	if !ok {
		return nil, nil, ErrGammaNotAchievable
	}
	vals := eig.Values(nil)
	g2 := gamma * gamma
	for _, v := range vals {
		if math.Abs(real(v)) >= g2 {
			return nil, nil, ErrGammaNotAchievable
		}
	}

	_ = D12
	_ = D21

	return X, Y, nil
}

func solveHamiltonianRiccati(H *mat.Dense, n int) (*mat.Dense, error) {
	nn := 2 * n
	hRaw := H.RawMatrix()
	hData := make([]float64, nn*nn)
	copyStrided(hData, nn, hRaw.Data, hRaw.Stride, nn, nn)

	wr := make([]float64, nn)
	wi := make([]float64, nn)
	vs := make([]float64, nn*nn)
	bwork := make([]bool, nn)

	selctg := func(wr, wi float64) bool { return wr < 0 }

	var workQuery [1]float64
	impl.Dgees(lapack.SchurHess, lapack.SortSelected, selctg,
		nn, hData, nn, wr, wi, vs, nn, workQuery[:], -1, bwork)
	lwork := int(workQuery[0])
	work := make([]float64, lwork)

	sdim, ok := impl.Dgees(lapack.SchurHess, lapack.SortSelected, selctg,
		nn, hData, nn, wr, wi, vs, nn, work, lwork, bwork)
	if !ok {
		return nil, ErrSchurFailed
	}
	if sdim != n {
		return nil, ErrNoStabilizing
	}

	u11 := make([]float64, n*n)
	u21 := make([]float64, n*n)
	copyStrided(u11, n, vs, nn, n, n)
	copyBlock(u21, n, 0, 0, vs, nn, n, 0, n, n)

	ipiv := make([]int, n)
	if !impl.Dgetrf(n, n, u11, n, ipiv) {
		return nil, ErrNoStabilizing
	}

	// X = U21 * U11^{-1}  =>  solve U11' * X' = U21'  =>  transpose approach
	xData := make([]float64, n*n)
	for i := range n {
		for j := range n {
			xData[i*n+j] = u21[j*n+i]
		}
	}
	impl.Dgetrs(blas.Trans, n, n, u11, n, ipiv, xData, n)
	symmetrize(xData, n, n)

	// Check positive semi-definiteness via eigenvalues
	X := mat.NewDense(n, n, xData)
	var eig mat.Eigen
	eigOk := eig.Factorize(X, mat.EigenNone)
	if !eigOk {
		return nil, ErrNoStabilizing
	}
	vals := eig.Values(nil)
	for _, v := range vals {
		if real(v) < -1e-8 {
			return nil, ErrNoStabilizing
		}
	}

	return X, nil
}

func maxSVD(M *mat.Dense) float64 {
	r, c := M.Dims()
	if r == 0 || c == 0 {
		return 0
	}
	raw := M.RawMatrix()
	allZero := true
	for i := range r {
		for j := range c {
			if raw.Data[i*raw.Stride+j] != 0 {
				allZero = false
				break
			}
		}
		if !allZero {
			break
		}
	}
	if allZero {
		return 0
	}

	mCopy := make([]float64, r*c)
	copyStrided(mCopy, c, raw.Data, raw.Stride, r, c)

	minDim := r
	if c < minDim {
		minDim = c
	}
	s := make([]float64, minDim)
	ldu := max(1, r)
	ldvt := max(1, c)

	var wkQuery [1]float64
	impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, r, c, mCopy, c, s, nil, ldu, nil, ldvt, wkQuery[:], -1)
	lwork := int(wkQuery[0])
	work := make([]float64, lwork)
	ok := impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, r, c, mCopy, c, s, nil, ldu, nil, ldvt, work, lwork)
	if !ok {
		return 0
	}

	maxS := 0.0
	for _, sv := range s {
		if sv > maxS {
			maxS = sv
		}
	}
	return maxS
}

