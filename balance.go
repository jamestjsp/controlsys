package controlsys

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/gonum/mat"
)

type BalredMethod int

const (
	Truncate             BalredMethod = iota
	SingularPerturbation
)

type BalrealResult struct {
	Sys  *System
	HSV  []float64
	T    *mat.Dense
	Tinv *mat.Dense
}

// Balreal computes the balanced realization of a stable, minimal LTI system.
//
// The returned system has equal controllability and observability gramians,
// both equal to diag(σ₁, σ₂, …, σₙ) where σᵢ are the Hankel singular values.
func Balreal(sys *System) (*BalrealResult, error) {
	n, m, p := sys.Dims()
	if n == 0 {
		cp := sys.Copy()
		return &BalrealResult{Sys: cp}, nil
	}

	stable, err := sys.IsStable()
	if err != nil {
		return nil, err
	}
	if !stable {
		return nil, ErrUnstable
	}

	aRaw := sys.A.RawMatrix()
	bRaw := sys.B.RawMatrix()
	cRaw := sys.C.RawMatrix()

	aData := make([]float64, n*n)
	at := make([]float64, n*n)
	copyStrided(aData, n, aRaw.Data, aRaw.Stride, n, n)
	for i := range n {
		for j := range n {
			at[i*n+j] = aRaw.Data[j*aRaw.Stride+i]
		}
	}

	bbt := make([]float64, n*n)
	blas64.Gemm(blas.NoTrans, blas.Trans, 1,
		blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
		blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: bbt})
	symmetrize(bbt, n, n)

	ctc := make([]float64, n*n)
	blas64.Gemm(blas.Trans, blas.NoTrans, 1,
		blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
		blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: ctc})
	symmetrize(ctc, n, n)

	A := mat.NewDense(n, n, aData)
	At := mat.NewDense(n, n, at)

	var Wc, Wo *mat.Dense
	if sys.IsContinuous() {
		Wc, err = Lyap(A, mat.NewDense(n, n, bbt))
		if err != nil {
			return nil, err
		}
		Wo, err = Lyap(At, mat.NewDense(n, n, ctc))
	} else {
		Wc, err = DLyap(A, mat.NewDense(n, n, bbt))
		if err != nil {
			return nil, err
		}
		Wo, err = DLyap(At, mat.NewDense(n, n, ctc))
	}
	if err != nil {
		return nil, err
	}

	wcRaw := Wc.RawMatrix()
	lc := make([]float64, n*n)
	copyStrided(lc, n, wcRaw.Data, wcRaw.Stride, n, n)
	if !impl.Dpotrf(blas.Lower, n, lc, n) {
		return nil, ErrNotMinimal
	}
	for i := range n {
		for j := i + 1; j < n; j++ {
			lc[i*n+j] = 0
		}
	}

	woRaw := Wo.RawMatrix()
	lo := make([]float64, n*n)
	copyStrided(lo, n, woRaw.Data, woRaw.Stride, n, n)
	if !impl.Dpotrf(blas.Lower, n, lo, n) {
		return nil, ErrNotMinimal
	}
	for i := range n {
		for j := i + 1; j < n; j++ {
			lo[i*n+j] = 0
		}
	}

	mData := make([]float64, n*n)
	lcGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: lc}
	loGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: lo}
	blas64.Gemm(blas.Trans, blas.NoTrans, 1, loGen, lcGen,
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: mData})

	s := make([]float64, n)
	uData := make([]float64, n*n)
	vtData := make([]float64, n*n)
	wq := make([]float64, 1)
	impl.Dgesvd(lapack.SVDAll, lapack.SVDAll, n, n, mData, n, s,
		uData, n, vtData, n, wq, -1)
	svdWork := make([]float64, int(wq[0]))
	impl.Dgesvd(lapack.SVDAll, lapack.SVDAll, n, n, mData, n, s,
		uData, n, vtData, n, svdWork, len(svdWork))

	// V = Vt' — scale columns of Vt' by 1/sqrt(σ) → vScaled
	// T = Lc * vScaled
	vScaled := make([]float64, n*n)
	for i := range n {
		invSqrt := 1.0 / math.Sqrt(s[i])
		for j := range n {
			vScaled[j*n+i] = vtData[i*n+j] * invSqrt
		}
	}

	tData := make([]float64, n*n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, lcGen,
		blas64.General{Rows: n, Cols: n, Stride: n, Data: vScaled},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: tData})
	T := mat.NewDense(n, n, tData)

	// Tinv = Σ^{-1/2} * U' * Lo'
	// Scale columns of U by 1/sqrt(σ) → uScaled, then Tinv = uScaled' * Lo'
	for j := range n {
		invSqrt := 1.0 / math.Sqrt(s[j])
		for i := range n {
			uData[i*n+j] *= invSqrt
		}
	}

	tinvData := make([]float64, n*n)
	blas64.Gemm(blas.Trans, blas.Trans, 1,
		blas64.General{Rows: n, Cols: n, Stride: n, Data: uData},
		loGen,
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: tinvData})
	Tinv := mat.NewDense(n, n, tinvData)

	tGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: tData}
	tiGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: tinvData}

	// Ab = Tinv * A * T
	tmp := make([]float64, n*n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, tiGen,
		blas64.General{Rows: n, Cols: n, Stride: aRaw.Stride, Data: aRaw.Data},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: tmp})
	abData := make([]float64, n*n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: n, Cols: n, Stride: n, Data: tmp},
		tGen,
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: abData})

	// Bb = Tinv * B
	bbData := make([]float64, n*m)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, tiGen,
		blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
		0, blas64.General{Rows: n, Cols: m, Stride: m, Data: bbData})

	// Cb = C * T
	cbData := make([]float64, p*n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
		tGen,
		0, blas64.General{Rows: p, Cols: n, Stride: n, Data: cbData})

	Ab := mat.NewDense(n, n, abData)
	Bb := mat.NewDense(n, m, bbData)
	Cb := mat.NewDense(p, n, cbData)
	Db := denseCopy(sys.D)

	balSys, _ := newNoCopy(Ab, Bb, Cb, Db, sys.Dt)
	propagateIONames(balSys, sys)

	return &BalrealResult{
		Sys:  balSys,
		HSV:  s,
		T:    T,
		Tinv: Tinv,
	}, nil
}

// Balred computes a reduced-order model via balanced truncation.
//
// If order is 0, the truncation order is automatically selected based on
// the largest relative gap in the Hankel singular values.
// method selects between Truncate (default) and SingularPerturbation (DC-gain matching).
func Balred(sys *System, order int, method BalredMethod) (*System, []float64, error) {
	br, err := Balreal(sys)
	if err != nil {
		return nil, nil, err
	}

	n, m, p := br.Sys.Dims()
	hsv := br.HSV

	r := order
	if r == 0 {
		r = autoSelectOrder(hsv)
	}
	if r < 1 || r >= n {
		return nil, nil, ErrInvalidOrder
	}

	Ab := br.Sys.A
	Bb := br.Sys.B
	Cb := br.Sys.C
	Db := br.Sys.D

	A11 := extractSubmatrix(Ab, 0, r, 0, r)
	B1 := extractSubmatrix(Bb, 0, r, 0, m)
	C1 := extractSubmatrix(Cb, 0, p, 0, r)

	if method == Truncate {
		Dr := denseCopy(Db)
		red, err := newNoCopy(A11, B1, C1, Dr, sys.Dt)
		if err != nil {
			return nil, hsv, err
		}
		propagateIONames(red, sys)
		return red, hsv, nil
	}

	red, hsvOut, err := singularPerturbation(Ab, Bb, Cb, Db, A11, B1, C1, n, m, p, r, hsv, sys.Dt)
	if err != nil {
		return nil, hsvOut, err
	}
	propagateIONames(red, sys)
	return red, hsvOut, nil
}

// Modred reduces a system by eliminating specified states.
//
// elim contains 0-based indices of states to remove.
// method selects Truncate or SingularPerturbation (DC-gain matching).
func Modred(sys *System, elim []int, method BalredMethod) (*System, error) {
	n, m, p := sys.Dims()
	if n == 0 || len(elim) == 0 {
		return sys.Copy(), nil
	}

	elimSet := make(map[int]bool, len(elim))
	for _, idx := range elim {
		if idx < 0 || idx >= n {
			return nil, ErrInvalidOrder
		}
		elimSet[idx] = true
	}
	if len(elimSet) != len(elim) {
		return nil, ErrInvalidOrder
	}

	keep := make([]int, 0, n-len(elim))
	for i := range n {
		if !elimSet[i] {
			keep = append(keep, i)
		}
	}
	r := len(keep)

	if r == 0 {
		g, err := NewGain(denseCopy(sys.D), sys.Dt)
		if err != nil {
			return nil, err
		}
		propagateIONames(g, sys)
		return g, nil
	}
	if r == n {
		return sys.Copy(), nil
	}

	perm := make([]int, n)
	copy(perm, keep)
	ei := 0
	for i := range n {
		if elimSet[i] {
			perm[r+ei] = i
			ei++
		}
	}

	aRaw := sys.A.RawMatrix()
	ap := make([]float64, n*n)
	for i := range n {
		for j := range n {
			ap[i*n+j] = aRaw.Data[perm[i]*aRaw.Stride+perm[j]]
		}
	}

	bRaw := sys.B.RawMatrix()
	bp := make([]float64, n*m)
	for i := range n {
		for j := range m {
			bp[i*m+j] = bRaw.Data[perm[i]*bRaw.Stride+j]
		}
	}

	cRaw := sys.C.RawMatrix()
	cp := make([]float64, p*n)
	for i := range p {
		for j := range n {
			cp[i*n+j] = cRaw.Data[i*cRaw.Stride+perm[j]]
		}
	}

	Ap := mat.NewDense(n, n, ap)
	Bp := mat.NewDense(n, m, bp)
	Cp := mat.NewDense(p, n, cp)

	A11 := extractSubmatrix(Ap, 0, r, 0, r)
	B1 := extractSubmatrix(Bp, 0, r, 0, m)
	C1 := extractSubmatrix(Cp, 0, p, 0, r)

	if method == Truncate {
		red, err := newNoCopy(A11, B1, C1, denseCopy(sys.D), sys.Dt)
		if err != nil {
			return nil, err
		}
		propagateIONames(red, sys)
		return red, nil
	}

	red, _, err := singularPerturbation(Ap, Bp, Cp, sys.D, A11, B1, C1, n, m, p, r, nil, sys.Dt)
	if err != nil {
		return nil, err
	}
	propagateIONames(red, sys)
	return red, nil
}

// singularPerturbation computes the reduced system via residualization.
// Ar = A11 - A12*inv(A22)*A21, etc. Uses Gemm beta=-1 to fuse multiply-subtract.
func singularPerturbation(
	Ab, Bb, Cb, Db *mat.Dense,
	A11, B1, C1 *mat.Dense,
	n, m, p, r int,
	hsv []float64, dt float64,
) (*System, []float64, error) {
	A12 := extractSubmatrix(Ab, 0, r, r, n)
	A21 := extractSubmatrix(Ab, r, n, 0, r)
	A22 := extractSubmatrix(Ab, r, n, r, n)
	B2 := extractSubmatrix(Bb, r, n, 0, m)
	C2 := extractSubmatrix(Cb, 0, p, r, n)

	n2 := n - r
	a22Data := make([]float64, n2*n2)
	a22Raw := A22.RawMatrix()
	copyStrided(a22Data, n2, a22Raw.Data, a22Raw.Stride, n2, n2)

	ipiv := make([]int, n2)
	if !impl.Dgetrf(n2, n2, a22Data, n2, ipiv) {
		return nil, hsv, ErrSingularA22
	}

	// Solve A22 * X = A21  and  A22 * Y = B2
	rhs1 := make([]float64, n2*r)
	a21Raw := A21.RawMatrix()
	copyStrided(rhs1, r, a21Raw.Data, a21Raw.Stride, n2, r)
	impl.Dgetrs(blas.NoTrans, n2, r, a22Data, n2, ipiv, rhs1, r)

	rhs2 := make([]float64, n2*m)
	b2Raw := B2.RawMatrix()
	copyStrided(rhs2, m, b2Raw.Data, b2Raw.Stride, n2, m)
	impl.Dgetrs(blas.NoTrans, n2, m, a22Data, n2, ipiv, rhs2, m)

	invA22_A21 := blas64.General{Rows: n2, Cols: r, Stride: r, Data: rhs1}
	invA22_B2 := blas64.General{Rows: n2, Cols: m, Stride: m, Data: rhs2}
	a12Raw := A12.RawMatrix()
	a12Gen := blas64.General{Rows: r, Cols: n2, Stride: a12Raw.Stride, Data: a12Raw.Data}
	c2Raw := C2.RawMatrix()
	c2Gen := blas64.General{Rows: p, Cols: n2, Stride: c2Raw.Stride, Data: c2Raw.Data}

	// Ar = A11 - A12 * inv(A22) * A21 → copy A11, then Gemm with beta=1, alpha=-1
	arData := make([]float64, r*r)
	a11Raw := A11.RawMatrix()
	copyStrided(arData, r, a11Raw.Data, a11Raw.Stride, r, r)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, a12Gen, invA22_A21,
		1, blas64.General{Rows: r, Cols: r, Stride: r, Data: arData})

	// Br = B1 - A12 * inv(A22) * B2
	brData := make([]float64, r*m)
	b1Raw := B1.RawMatrix()
	copyStrided(brData, m, b1Raw.Data, b1Raw.Stride, r, m)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, a12Gen, invA22_B2,
		1, blas64.General{Rows: r, Cols: m, Stride: m, Data: brData})

	// Cr = C1 - C2 * inv(A22) * A21
	crData := make([]float64, p*r)
	c1Raw := C1.RawMatrix()
	copyStrided(crData, r, c1Raw.Data, c1Raw.Stride, p, r)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, c2Gen, invA22_A21,
		1, blas64.General{Rows: p, Cols: r, Stride: r, Data: crData})

	// Dr = D - C2 * inv(A22) * B2
	drData := make([]float64, p*m)
	dRaw := Db.RawMatrix()
	copyStrided(drData, m, dRaw.Data, dRaw.Stride, p, m)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, c2Gen, invA22_B2,
		1, blas64.General{Rows: p, Cols: m, Stride: m, Data: drData})

	red, err := newNoCopy(
		mat.NewDense(r, r, arData),
		mat.NewDense(r, m, brData),
		mat.NewDense(p, r, crData),
		mat.NewDense(p, m, drData),
		dt,
	)
	return red, hsv, err
}

func autoSelectOrder(hsv []float64) int {
	if len(hsv) <= 1 {
		return len(hsv)
	}
	bestIdx := 0
	bestRatio := 0.0
	for i := 0; i < len(hsv)-1; i++ {
		if hsv[i+1] > 0 {
			ratio := hsv[i] / hsv[i+1]
			if ratio > bestRatio {
				bestRatio = ratio
				bestIdx = i
			}
		} else if hsv[i] > 0 {
			bestIdx = i
			break
		}
	}
	return bestIdx + 1
}
