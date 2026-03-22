package controlsys

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/gonum/mat"
)

type RiccatiOpts struct {
	S *mat.Dense
}

type RiccatiResult struct {
	X    *mat.Dense
	K    *mat.Dense
	Eig  []complex128
	Rcnd float64
}

// Care solves the continuous algebraic Riccati equation:
//
//	A'X + XA - (XB+S)*R⁻¹*(B'X+S') + Q = 0
//
// When opts is nil or opts.S is nil, the cross-term is zero:
//
//	A'X + XA - XB*R⁻¹*B'X + Q = 0
//
// A is n×n, B is n×m, Q is n×n symmetric, R is m×m symmetric positive definite.
func Care(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	na, nac := A.Dims()
	if na != nac {
		return nil, ErrDimensionMismatch
	}
	nb, m := B.Dims()
	if nb != na {
		return nil, ErrDimensionMismatch
	}
	qr, qc := Q.Dims()
	if qr != na || qc != na {
		return nil, ErrDimensionMismatch
	}
	rr, rc := R.Dims()
	if rr != m || rc != m {
		return nil, ErrDimensionMismatch
	}
	n := na
	if n == 0 {
		return &RiccatiResult{X: &mat.Dense{}, K: &mat.Dense{}, Eig: nil}, nil
	}
	if !isSymmetric(Q, eps()*denseNorm(Q)) {
		return nil, ErrNotSymmetric
	}
	if !isSymmetric(R, eps()*denseNorm(R)) {
		return nil, ErrNotSymmetric
	}

	var S *mat.Dense
	if opts != nil && opts.S != nil {
		sr, sc := opts.S.Dims()
		if sr != n || sc != m {
			return nil, ErrDimensionMismatch
		}
		S = opts.S
	}

	// Cholesky factor R
	rChol := make([]float64, m*m)
	rRaw := R.RawMatrix()
	for i := range m {
		copy(rChol[i*m:i*m+m], rRaw.Data[i*rRaw.Stride:i*rRaw.Stride+m])
	}
	if !impl.Dpotrf(blas.Upper, m, rChol, m) {
		return nil, ErrSingularR
	}

	// Working copies of A and Q for cross-term transformation
	aWork := make([]float64, n*n)
	aRaw := A.RawMatrix()
	for i := range n {
		copy(aWork[i*n:i*n+n], aRaw.Data[i*aRaw.Stride:i*aRaw.Stride+n])
	}
	qWork := make([]float64, n*n)
	qRaw := Q.RawMatrix()
	for i := range n {
		copy(qWork[i*n:i*n+n], qRaw.Data[i*qRaw.Stride:i*qRaw.Stride+n])
	}

	// Compute R⁻¹*B' (m×n): solve R*W = B' via Dpotrs
	rinvBt := make([]float64, m*n)
	bRaw := B.RawMatrix()
	for i := range n {
		for j := range m {
			rinvBt[j*n+i] = bRaw.Data[i*bRaw.Stride+j]
		}
	}
	impl.Dpotrs(blas.Upper, m, n, rChol, m, rinvBt, n)

	if S != nil {
		// Ã = A - B*R⁻¹*S'
		// Z = R⁻¹*S' (m×n): solve R*Z = S'
		rinvSt := make([]float64, m*n)
		sRaw := S.RawMatrix()
		for i := range n {
			for j := range m {
				rinvSt[j*n+i] = sRaw.Data[i*sRaw.Stride+j]
			}
		}
		impl.Dpotrs(blas.Upper, m, n, rChol, m, rinvSt, n)

		// aWork -= B * Z
		blas64.Gemm(blas.NoTrans, blas.NoTrans,
			-1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
			blas64.General{Rows: m, Cols: n, Data: rinvSt, Stride: n},
			1, blas64.General{Rows: n, Cols: n, Data: aWork, Stride: n})

		// qWork -= S * Z
		blas64.Gemm(blas.NoTrans, blas.NoTrans,
			-1, blas64.General{Rows: n, Cols: m, Data: sRaw.Data, Stride: sRaw.Stride},
			blas64.General{Rows: m, Cols: n, Data: rinvSt, Stride: n},
			1, blas64.General{Rows: n, Cols: n, Data: qWork, Stride: n})
		symmetrize(qWork, n, n)
	}

	// G = B * R⁻¹ * B' (n×n symmetric)
	g := make([]float64, n*n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		blas64.General{Rows: m, Cols: n, Data: rinvBt, Stride: n},
		0, blas64.General{Rows: n, Cols: n, Data: g, Stride: n})
	symmetrize(g, n, n)

	// Form 2n×2n Hamiltonian: H = [[A, -G], [-Q, -A']]
	nn := 2 * n
	h := make([]float64, nn*nn)
	for i := range n {
		for j := range n {
			h[i*nn+j] = aWork[i*n+j]
			h[i*nn+n+j] = -g[i*n+j]
			h[(n+i)*nn+j] = -qWork[i*n+j]
			h[(n+i)*nn+n+j] = -aWork[j*n+i]
		}
	}

	// Schur decomposition with sorting: Re(λ) < 0 to top-left
	wr := make([]float64, nn)
	wi := make([]float64, nn)
	vs := make([]float64, nn*nn)
	bwork := make([]bool, nn)

	selctg := func(wr, wi float64) bool { return wr < 0 }

	workQuery := make([]float64, 1)
	impl.Dgees(lapack.SchurHess, lapack.SortSelected, selctg,
		nn, h, nn, wr, wi, vs, nn, workQuery, -1, bwork)
	lwork := int(workQuery[0])
	work := make([]float64, lwork)

	sdim, ok := impl.Dgees(lapack.SchurHess, lapack.SortSelected, selctg,
		nn, h, nn, wr, wi, vs, nn, work, lwork, bwork)
	if !ok {
		return nil, ErrSchurFailed
	}
	if sdim != n {
		return nil, ErrNoStabilizing
	}

	// Extract U11 = vs[0:n, 0:n], U21 = vs[n:2n, 0:n]
	u11 := make([]float64, n*n)
	u21 := make([]float64, n*n)
	for i := range n {
		copy(u11[i*n:i*n+n], vs[i*nn:i*nn+n])
		copy(u21[i*n:i*n+n], vs[(n+i)*nn:(n+i)*nn+n])
	}

	// X = U21 * U11⁻¹; since X is symmetric: X = (U11')⁻¹ * U21'
	// Solve via DGETRS(Trans) instead of explicit inverse
	ipiv := make([]int, n)
	if !impl.Dgetrf(n, n, u11, n, ipiv) {
		return nil, ErrNoStabilizing
	}

	anorm := impl.Dlange(lapack.MaxColumnSum, n, n, u11, n, work[:n])
	rcnd := impl.Dgecon(lapack.MaxColumnSum, n, u11, n, anorm, work[:4*n], make([]int, n))

	xData := make([]float64, n*n)
	for i := range n {
		for j := range n {
			xData[i*n+j] = u21[j*n+i]
		}
	}
	impl.Dgetrs(blas.Trans, n, n, u11, n, ipiv, xData, n)

	symmetrize(xData, n, n)
	X := mat.NewDense(n, n, xData)

	// Closed-loop eigenvalues
	eig := make([]complex128, n)
	for i := range n {
		eig[i] = complex(wr[i], wi[i])
	}

	// Gain K = R⁻¹ * (B'X + S')
	// kData = B'*X (m×n)
	kData := make([]float64, m*n)
	blas64.Gemm(blas.Trans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		blas64.General{Rows: n, Cols: n, Data: xData, Stride: n},
		0, blas64.General{Rows: m, Cols: n, Data: kData, Stride: n})
	if S != nil {
		sRaw := S.RawMatrix()
		for i := range n {
			for j := range m {
				kData[j*n+i] += sRaw.Data[i*sRaw.Stride+j]
			}
		}
	}
	impl.Dpotrs(blas.Upper, m, n, rChol, m, kData, n)
	K := mat.NewDense(m, n, kData)

	return &RiccatiResult{X: X, K: K, Eig: eig, Rcnd: rcnd}, nil
}

// Dare solves the discrete algebraic Riccati equation:
//
//	A'XA - X - (A'XB+S)*(R+B'XB)⁻¹*(B'XA+S') + Q = 0
//
// When opts is nil or opts.S is nil, the cross-term is zero:
//
//	A'XA - X - A'XB*(R+B'XB)⁻¹*B'XA + Q = 0
//
// A is n×n, B is n×m, Q is n×n symmetric, R is m×m symmetric positive definite.
func Dare(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	na, nac := A.Dims()
	if na != nac {
		return nil, ErrDimensionMismatch
	}
	nb, m := B.Dims()
	if nb != na {
		return nil, ErrDimensionMismatch
	}
	qr, qc := Q.Dims()
	if qr != na || qc != na {
		return nil, ErrDimensionMismatch
	}
	rr, rc := R.Dims()
	if rr != m || rc != m {
		return nil, ErrDimensionMismatch
	}
	n := na
	if n == 0 {
		return &RiccatiResult{X: &mat.Dense{}, K: &mat.Dense{}, Eig: nil}, nil
	}
	if !isSymmetric(Q, eps()*denseNorm(Q)) {
		return nil, ErrNotSymmetric
	}
	if !isSymmetric(R, eps()*denseNorm(R)) {
		return nil, ErrNotSymmetric
	}

	var S *mat.Dense
	if opts != nil && opts.S != nil {
		sr, sc := opts.S.Dims()
		if sr != n || sc != m {
			return nil, ErrDimensionMismatch
		}
		S = opts.S
	}

	// Cholesky factor R
	rChol := make([]float64, m*m)
	rRaw := R.RawMatrix()
	for i := range m {
		copy(rChol[i*m:i*m+m], rRaw.Data[i*rRaw.Stride:i*rRaw.Stride+m])
	}
	if !impl.Dpotrf(blas.Upper, m, rChol, m) {
		return nil, ErrSingularR
	}

	aWork := make([]float64, n*n)
	aRaw := A.RawMatrix()
	for i := range n {
		copy(aWork[i*n:i*n+n], aRaw.Data[i*aRaw.Stride:i*aRaw.Stride+n])
	}
	qWork := make([]float64, n*n)
	qRaw := Q.RawMatrix()
	for i := range n {
		copy(qWork[i*n:i*n+n], qRaw.Data[i*qRaw.Stride:i*qRaw.Stride+n])
	}

	bRaw := B.RawMatrix()

	// R⁻¹*B' (m×n)
	rinvBt := make([]float64, m*n)
	for i := range n {
		for j := range m {
			rinvBt[j*n+i] = bRaw.Data[i*bRaw.Stride+j]
		}
	}
	impl.Dpotrs(blas.Upper, m, n, rChol, m, rinvBt, n)

	if S != nil {
		rinvSt := make([]float64, m*n)
		sRaw := S.RawMatrix()
		for i := range n {
			for j := range m {
				rinvSt[j*n+i] = sRaw.Data[i*sRaw.Stride+j]
			}
		}
		impl.Dpotrs(blas.Upper, m, n, rChol, m, rinvSt, n)

		blas64.Gemm(blas.NoTrans, blas.NoTrans,
			-1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
			blas64.General{Rows: m, Cols: n, Data: rinvSt, Stride: n},
			1, blas64.General{Rows: n, Cols: n, Data: aWork, Stride: n})

		blas64.Gemm(blas.NoTrans, blas.NoTrans,
			-1, blas64.General{Rows: n, Cols: m, Data: sRaw.Data, Stride: sRaw.Stride},
			blas64.General{Rows: m, Cols: n, Data: rinvSt, Stride: n},
			1, blas64.General{Rows: n, Cols: n, Data: qWork, Stride: n})
		symmetrize(qWork, n, n)
	}

	// G = B * R⁻¹ * B'
	g := make([]float64, n*n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		blas64.General{Rows: m, Cols: n, Data: rinvBt, Stride: n},
		0, blas64.General{Rows: n, Cols: n, Data: g, Stride: n})
	symmetrize(g, n, n)

	// Form symplectic matrix Z = M⁻¹*L (requires A invertible):
	// Z = [[A + G*Ait*Q, -G*Ait], [-Ait*Q, Ait]]  where Ait = (A')⁻¹
	// Compute Ait = (A')⁻¹ via LU factorization of A'
	ait := make([]float64, n*n)
	for i := range n {
		for j := range n {
			ait[i*n+j] = aWork[j*n+i]
		}
	}
	aipiv := make([]int, n)
	if !impl.Dgetrf(n, n, ait, n, aipiv) {
		return nil, ErrNoStabilizing
	}
	aiWorkQ := make([]float64, 1)
	impl.Dgetri(n, ait, n, aipiv, aiWorkQ, -1)
	aiWork := make([]float64, int(aiWorkQ[0]))
	impl.Dgetri(n, ait, n, aipiv, aiWork, len(aiWork))

	// Ait*Q (n×n)
	aitq := make([]float64, n*n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: n, Data: ait, Stride: n},
		blas64.General{Rows: n, Cols: n, Data: qWork, Stride: n},
		0, blas64.General{Rows: n, Cols: n, Data: aitq, Stride: n})

	// G*Ait (n×n)
	gait := make([]float64, n*n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: n, Data: g, Stride: n},
		blas64.General{Rows: n, Cols: n, Data: ait, Stride: n},
		0, blas64.General{Rows: n, Cols: n, Data: gait, Stride: n})

	gaitq := make([]float64, n*n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: n, Data: gait, Stride: n},
		blas64.General{Rows: n, Cols: n, Data: qWork, Stride: n},
		0, blas64.General{Rows: n, Cols: n, Data: gaitq, Stride: n})

	nn := 2 * n
	z := make([]float64, nn*nn)
	for i := range n {
		for j := range n {
			z[i*nn+j] = aWork[i*n+j] + gaitq[i*n+j]
			z[i*nn+n+j] = -gait[i*n+j]
			z[(n+i)*nn+j] = -aitq[i*n+j]
			z[(n+i)*nn+n+j] = ait[i*n+j]
		}
	}

	// Schur decomposition with sorting: |λ| < 1 to top-left
	wr := make([]float64, nn)
	wi := make([]float64, nn)
	vs := make([]float64, nn*nn)
	bwork := make([]bool, nn)

	selctg := func(wr, wi float64) bool {
		return wr*wr+wi*wi < 1
	}

	workQuery := make([]float64, 1)
	impl.Dgees(lapack.SchurHess, lapack.SortSelected, selctg,
		nn, z, nn, wr, wi, vs, nn, workQuery, -1, bwork)
	lwork := int(workQuery[0])
	work := make([]float64, lwork)

	sdim, ok := impl.Dgees(lapack.SchurHess, lapack.SortSelected, selctg,
		nn, z, nn, wr, wi, vs, nn, work, lwork, bwork)
	if !ok {
		return nil, ErrSchurFailed
	}
	if sdim != n {
		return nil, ErrNoStabilizing
	}

	// Extract U11 = vs[0:n, 0:n], U21 = vs[n:2n, 0:n]
	u11 := make([]float64, n*n)
	u21 := make([]float64, n*n)
	for i := range n {
		copy(u11[i*n:i*n+n], vs[i*nn:i*nn+n])
		copy(u21[i*n:i*n+n], vs[(n+i)*nn:(n+i)*nn+n])
	}

	ipiv := make([]int, n)
	if !impl.Dgetrf(n, n, u11, n, ipiv) {
		return nil, ErrNoStabilizing
	}

	anorm := impl.Dlange(lapack.MaxColumnSum, n, n, u11, n, work[:n])
	rcnd := impl.Dgecon(lapack.MaxColumnSum, n, u11, n, anorm, work[:4*n], make([]int, n))

	xData := make([]float64, n*n)
	for i := range n {
		for j := range n {
			xData[i*n+j] = u21[j*n+i]
		}
	}
	impl.Dgetrs(blas.Trans, n, n, u11, n, ipiv, xData, n)

	symmetrize(xData, n, n)
	X := mat.NewDense(n, n, xData)

	// Closed-loop eigenvalues
	eig := make([]complex128, n)
	for i := range n {
		eig[i] = complex(wr[i], wi[i])
	}

	// Gain K = (R + B'XB)⁻¹ * (B'XA + S')
	btx := make([]float64, m*n)
	blas64.Gemm(blas.Trans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		blas64.General{Rows: n, Cols: n, Data: xData, Stride: n},
		0, blas64.General{Rows: m, Cols: n, Data: btx, Stride: n})

	rbar := make([]float64, m*m)
	for i := range m {
		copy(rbar[i*m:i*m+m], rRaw.Data[i*rRaw.Stride:i*rRaw.Stride+m])
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: m, Cols: n, Data: btx, Stride: n},
		blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		1, blas64.General{Rows: m, Cols: m, Data: rbar, Stride: m})

	if !impl.Dpotrf(blas.Upper, m, rbar, m) {
		return nil, ErrSingularR
	}

	// BtXA = BtX * A (m×n)
	kData := make([]float64, m*n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: m, Cols: n, Data: btx, Stride: n},
		blas64.General{Rows: n, Cols: n, Data: aRaw.Data, Stride: aRaw.Stride},
		0, blas64.General{Rows: m, Cols: n, Data: kData, Stride: n})
	if S != nil {
		sRaw := S.RawMatrix()
		for i := range n {
			for j := range m {
				kData[j*n+i] += sRaw.Data[i*sRaw.Stride+j]
			}
		}
	}
	impl.Dpotrs(blas.Upper, m, n, rbar, m, kData, n)
	K := mat.NewDense(m, n, kData)

	return &RiccatiResult{X: X, K: K, Eig: eig, Rcnd: rcnd}, nil
}
