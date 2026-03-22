package controlsys

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
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
	aData := make([]float64, n*n)
	at := make([]float64, n*n)
	for i := range n {
		for j := range n {
			aData[i*n+j] = aRaw.Data[i*aRaw.Stride+j]
			at[i*n+j] = aRaw.Data[j*aRaw.Stride+i]
		}
	}

	bbt := make([]float64, n*n)
	bRaw := sys.B.RawMatrix()
	blas64.Gemm(blas.NoTrans, blas.Trans, 1,
		blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
		blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: bbt})
	symmetrize(bbt, n, n)

	ctc := make([]float64, n*n)
	cRaw := sys.C.RawMatrix()
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
	for i := range n {
		copy(lc[i*n:i*n+n], wcRaw.Data[i*wcRaw.Stride:i*wcRaw.Stride+n])
	}
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
	for i := range n {
		copy(lo[i*n:i*n+n], woRaw.Data[i*woRaw.Stride:i*woRaw.Stride+n])
	}
	if !impl.Dpotrf(blas.Lower, n, lo, n) {
		return nil, ErrNotMinimal
	}
	for i := range n {
		for j := i + 1; j < n; j++ {
			lo[i*n+j] = 0
		}
	}

	mData := make([]float64, n*n)
	blas64.Gemm(blas.Trans, blas.NoTrans, 1,
		blas64.General{Rows: n, Cols: n, Stride: n, Data: lo},
		blas64.General{Rows: n, Cols: n, Stride: n, Data: lc},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: mData})

	M := mat.NewDense(n, n, mData)
	var svd mat.SVD
	if !svd.Factorize(M, mat.SVDFull) {
		return nil, ErrSchurFailed
	}
	hsv := svd.Values(nil)
	var U, V mat.Dense
	svd.UTo(&U)
	svd.VTo(&V)

	vScaled := mat.DenseCopyOf(&V)
	vRaw := vScaled.RawMatrix()
	for j := range n {
		s := 1.0 / math.Sqrt(hsv[j])
		for i := range n {
			vRaw.Data[i*vRaw.Stride+j] *= s
		}
	}

	T := mat.NewDense(n, n, nil)
	lcDense := mat.NewDense(n, n, lc)
	T.Mul(lcDense, vScaled)

	uScaled := mat.DenseCopyOf(&U)
	uRaw := uScaled.RawMatrix()
	for j := range n {
		s := 1.0 / math.Sqrt(hsv[j])
		for i := range n {
			uRaw.Data[i*uRaw.Stride+j] *= s
		}
	}

	Tinv := mat.NewDense(n, n, nil)
	loDense := mat.NewDense(n, n, lo)
	Tinv.Mul(uScaled.T(), loDense.T())

	tmp := mat.NewDense(n, n, nil)
	tmp.Mul(Tinv, sys.A)
	Ab := mat.NewDense(n, n, nil)
	Ab.Mul(tmp, T)

	Bb := mat.NewDense(n, m, nil)
	Bb.Mul(Tinv, sys.B)

	Cb := mat.NewDense(p, n, nil)
	Cb.Mul(sys.C, T)

	Db := denseCopy(sys.D)

	balSys, _ := newNoCopy(Ab, Bb, Cb, Db, sys.Dt)

	return &BalrealResult{
		Sys:  balSys,
		HSV:  hsv,
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
		return red, hsv, err
	}

	A12 := extractSubmatrix(Ab, 0, r, r, n)
	A21 := extractSubmatrix(Ab, r, n, 0, r)
	A22 := extractSubmatrix(Ab, r, n, r, n)
	B2 := extractSubmatrix(Bb, r, n, 0, m)
	C2 := extractSubmatrix(Cb, 0, p, r, n)

	n2 := n - r
	a22Data := make([]float64, n2*n2)
	a22Raw := A22.RawMatrix()
	for i := range n2 {
		copy(a22Data[i*n2:i*n2+n2], a22Raw.Data[i*a22Raw.Stride:i*a22Raw.Stride+n2])
	}

	ipiv := make([]int, n2)
	if !impl.Dgetrf(n2, n2, a22Data, n2, ipiv) {
		return nil, hsv, ErrSingularA22
	}

	rhs1 := make([]float64, n2*r)
	a21Raw := A21.RawMatrix()
	for i := range n2 {
		copy(rhs1[i*r:i*r+r], a21Raw.Data[i*a21Raw.Stride:i*a21Raw.Stride+r])
	}
	impl.Dgetrs(blas.NoTrans, n2, r, a22Data, n2, ipiv, rhs1, r)

	rhs2 := make([]float64, n2*m)
	b2Raw := B2.RawMatrix()
	for i := range n2 {
		copy(rhs2[i*m:i*m+m], b2Raw.Data[i*b2Raw.Stride:i*b2Raw.Stride+m])
	}
	impl.Dgetrs(blas.NoTrans, n2, m, a22Data, n2, ipiv, rhs2, m)

	invA22_A21 := mat.NewDense(n2, r, rhs1)
	invA22_B2 := mat.NewDense(n2, m, rhs2)

	Ar := mat.NewDense(r, r, nil)
	Ar.Mul(A12, invA22_A21)
	arRaw := Ar.RawMatrix()
	a11Raw := A11.RawMatrix()
	for i := range r {
		for j := range r {
			arRaw.Data[i*arRaw.Stride+j] = a11Raw.Data[i*a11Raw.Stride+j] - arRaw.Data[i*arRaw.Stride+j]
		}
	}

	Br := mat.NewDense(r, m, nil)
	Br.Mul(A12, invA22_B2)
	brRaw := Br.RawMatrix()
	b1Raw := B1.RawMatrix()
	for i := range r {
		for j := range m {
			brRaw.Data[i*brRaw.Stride+j] = b1Raw.Data[i*b1Raw.Stride+j] - brRaw.Data[i*brRaw.Stride+j]
		}
	}

	Cr := mat.NewDense(p, r, nil)
	Cr.Mul(C2, invA22_A21)
	crRaw := Cr.RawMatrix()
	c1Raw := C1.RawMatrix()
	for i := range p {
		for j := range r {
			crRaw.Data[i*crRaw.Stride+j] = c1Raw.Data[i*c1Raw.Stride+j] - crRaw.Data[i*crRaw.Stride+j]
		}
	}

	Dr := mat.NewDense(p, m, nil)
	Dr.Mul(C2, invA22_B2)
	drRaw := Dr.RawMatrix()
	dRaw := Db.RawMatrix()
	for i := range p {
		for j := range m {
			drRaw.Data[i*drRaw.Stride+j] = dRaw.Data[i*dRaw.Stride+j] - drRaw.Data[i*drRaw.Stride+j]
		}
	}

	red, err := newNoCopy(Ar, Br, Cr, Dr, sys.Dt)
	return red, hsv, err
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
		return NewGain(denseCopy(sys.D), sys.Dt)
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
		return newNoCopy(A11, B1, C1, denseCopy(sys.D), sys.Dt)
	}

	A12 := extractSubmatrix(Ap, 0, r, r, n)
	A21 := extractSubmatrix(Ap, r, n, 0, r)
	A22 := extractSubmatrix(Ap, r, n, r, n)
	B2 := extractSubmatrix(Bp, r, n, 0, m)
	C2 := extractSubmatrix(Cp, 0, p, r, n)

	n2 := n - r
	a22Data := make([]float64, n2*n2)
	a22Raw := A22.RawMatrix()
	for i := range n2 {
		copy(a22Data[i*n2:i*n2+n2], a22Raw.Data[i*a22Raw.Stride:i*a22Raw.Stride+n2])
	}
	ipiv := make([]int, n2)
	if !impl.Dgetrf(n2, n2, a22Data, n2, ipiv) {
		return nil, ErrSingularA22
	}

	rhs1 := make([]float64, n2*r)
	a21Raw := A21.RawMatrix()
	for i := range n2 {
		copy(rhs1[i*r:i*r+r], a21Raw.Data[i*a21Raw.Stride:i*a21Raw.Stride+r])
	}
	impl.Dgetrs(blas.NoTrans, n2, r, a22Data, n2, ipiv, rhs1, r)

	rhs2 := make([]float64, n2*m)
	b2Raw := B2.RawMatrix()
	for i := range n2 {
		copy(rhs2[i*m:i*m+m], b2Raw.Data[i*b2Raw.Stride:i*b2Raw.Stride+m])
	}
	impl.Dgetrs(blas.NoTrans, n2, m, a22Data, n2, ipiv, rhs2, m)

	inv22_21 := mat.NewDense(n2, r, rhs1)
	inv22_B2 := mat.NewDense(n2, m, rhs2)

	Ar := mat.NewDense(r, r, nil)
	Ar.Mul(A12, inv22_21)
	arRaw := Ar.RawMatrix()
	a11Raw := A11.RawMatrix()
	for i := range r {
		for j := range r {
			arRaw.Data[i*arRaw.Stride+j] = a11Raw.Data[i*a11Raw.Stride+j] - arRaw.Data[i*arRaw.Stride+j]
		}
	}

	Br := mat.NewDense(r, m, nil)
	Br.Mul(A12, inv22_B2)
	brRed := Br.RawMatrix()
	b1Raw := B1.RawMatrix()
	for i := range r {
		for j := range m {
			brRed.Data[i*brRed.Stride+j] = b1Raw.Data[i*b1Raw.Stride+j] - brRed.Data[i*brRed.Stride+j]
		}
	}

	Cr := mat.NewDense(p, r, nil)
	Cr.Mul(C2, inv22_21)
	crRaw := Cr.RawMatrix()
	c1Raw := C1.RawMatrix()
	for i := range p {
		for j := range r {
			crRaw.Data[i*crRaw.Stride+j] = c1Raw.Data[i*c1Raw.Stride+j] - crRaw.Data[i*crRaw.Stride+j]
		}
	}

	Dr := mat.NewDense(p, m, nil)
	Dr.Mul(C2, inv22_B2)
	drRaw := Dr.RawMatrix()
	dRaw := sys.D.RawMatrix()
	for i := range p {
		for j := range m {
			drRaw.Data[i*drRaw.Stride+j] = dRaw.Data[i*dRaw.Stride+j] - drRaw.Data[i*drRaw.Stride+j]
		}
	}

	return newNoCopy(Ar, Br, Cr, Dr, sys.Dt)
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

