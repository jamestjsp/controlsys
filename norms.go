package controlsys

import (
	"math"
	"math/cmplx"
	"sort"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/gonum/mat"
)

// H2Norm computes the H2 norm of a stable LTI system.
//
// For continuous systems with D ≠ 0, the H2 norm is infinite.
func H2Norm(sys *System) (float64, error) {
	n, m, p := sys.Dims()

	if n == 0 {
		if sys.IsContinuous() {
			return math.Inf(1), nil
		}
		return frobNormD(sys.D, p, m), nil
	}

	stable, err := sys.IsStable()
	if err != nil {
		return 0, err
	}
	if !stable {
		return 0, ErrUnstable
	}

	if sys.IsContinuous() && !allZeroDense(sys.D) {
		return math.Inf(1), nil
	}

	ctc := make([]float64, n*n)
	cRaw := sys.C.RawMatrix()
	blas64.Gemm(blas.Trans, blas.NoTrans, 1,
		blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
		blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: ctc})
	symmetrize(ctc, n, n)

	aRaw := sys.A.RawMatrix()
	at := make([]float64, n*n)
	for i := range n {
		for j := range n {
			at[i*n+j] = aRaw.Data[j*aRaw.Stride+i]
		}
	}
	At := mat.NewDense(n, n, at)
	Q := mat.NewDense(n, n, ctc)

	var X *mat.Dense
	if sys.IsContinuous() {
		X, err = Lyap(At, Q)
	} else {
		X, err = DLyap(At, Q)
	}
	if err != nil {
		return 0, err
	}

	xb := make([]float64, n*m)
	xRaw := X.RawMatrix()
	bRaw := sys.B.RawMatrix()
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: n, Cols: n, Stride: xRaw.Stride, Data: xRaw.Data},
		blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
		0, blas64.General{Rows: n, Cols: m, Stride: m, Data: xb})

	tr := 0.0
	for i := range n {
		for j := range m {
			tr += bRaw.Data[i*bRaw.Stride+j] * xb[i*m+j]
		}
	}

	if sys.IsDiscrete() {
		dRaw := sys.D.RawMatrix()
		for i := range p {
			for j := range m {
				v := dRaw.Data[i*dRaw.Stride+j]
				tr += v * v
			}
		}
	}

	if tr < 0 {
		tr = 0
	}
	return math.Sqrt(tr), nil
}

// HSV computes the Hankel singular values of a stable LTI system in descending order.
func HSV(sys *System) ([]float64, error) {
	n, m, p := sys.Dims()
	if n == 0 {
		return nil, nil
	}

	stable, err := sys.IsStable()
	if err != nil {
		return nil, err
	}
	if !stable {
		return nil, ErrUnstable
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

	aRaw := sys.A.RawMatrix()
	aData := make([]float64, n*n)
	at := make([]float64, n*n)
	for i := range n {
		for j := range n {
			aData[i*n+j] = aRaw.Data[i*aRaw.Stride+j]
			at[i*n+j] = aRaw.Data[j*aRaw.Stride+i]
		}
	}

	A := mat.NewDense(n, n, aData)
	At := mat.NewDense(n, n, at)
	BBt := mat.NewDense(n, n, bbt)
	CtC := mat.NewDense(n, n, ctc)

	var Wc, Wo *mat.Dense
	if sys.IsContinuous() {
		Wc, err = Lyap(A, BBt)
		if err != nil {
			return nil, err
		}
		Wo, err = Lyap(At, CtC)
	} else {
		Wc, err = DLyap(A, BBt)
		if err != nil {
			return nil, err
		}
		Wo, err = DLyap(At, CtC)
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
		return eigenvalueHSV(Wc, Wo, n), nil
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
		return eigenvalueHSV(Wc, Wo, n), nil
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

	s := make([]float64, n)
	wq := make([]float64, 1)
	impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, n, n, mData, n, s, nil, 1, nil, 1, wq, -1)
	work := make([]float64, int(wq[0]))
	impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, n, n, mData, n, s, nil, 1, nil, 1, work, len(work))

	return s, nil
}

func eigenvalueHSV(Wc, Wo *mat.Dense, n int) []float64 {
	prod := mat.NewDense(n, n, nil)
	prod.Mul(Wc, Wo)

	var eig mat.Eigen
	ok := eig.Factorize(prod, mat.EigenNone)
	if !ok {
		return make([]float64, n)
	}
	vals := eig.Values(nil)
	hsv := make([]float64, n)
	for i, v := range vals {
		r := real(v)
		if r < 0 {
			r = 0
		}
		hsv[i] = math.Sqrt(r)
	}
	sort.Sort(sort.Reverse(sort.Float64Slice(hsv)))
	return hsv
}

// HinfNorm computes the H∞ norm (peak gain) of a stable LTI system
// and the frequency at which it occurs.
func HinfNorm(sys *System) (norm float64, omega float64, err error) {
	n, m, p := sys.Dims()

	if n == 0 {
		sv := maxSVDense(sys.D, p, m)
		return sv, 0, nil
	}

	stable, err := sys.IsStable()
	if err != nil {
		return 0, 0, err
	}
	if !stable {
		return 0, 0, ErrUnstable
	}

	if sys.IsDiscrete() {
		csys, err := sys.Undiscretize()
		if err != nil {
			return 0, 0, err
		}
		norm, omega, err = HinfNorm(csys)
		return norm, omega, err
	}

	gammaLow, omegaPeak := hinfLowerBound(sys, m, p)

	gammaHigh := math.Max(gammaLow*2, 1e-10)
	for range 50 {
		if !hamiltonianHasImagEigs(sys, gammaHigh, n, m, p) {
			break
		}
		gammaHigh *= 2
	}

	tol := 1e-10
	for range 100 {
		if gammaHigh-gammaLow < tol*gammaHigh {
			break
		}
		mid := (gammaLow + gammaHigh) / 2
		if hamiltonianHasImagEigs(sys, mid, n, m, p) {
			gammaLow = mid
		} else {
			gammaHigh = mid
		}
	}

	return gammaHigh, omegaPeak, nil
}

func hinfLowerBound(sys *System, m, p int) (gammaLow, omegaPeak float64) {
	poles, err := sys.Poles()
	if err != nil {
		return 0, 0
	}

	freqs := make([]float64, 0, 60)
	freqs = append(freqs, 0)

	wmin, wmax := math.Inf(1), 0.0
	for _, pole := range poles {
		w := cmplx.Abs(pole)
		if w > 0 {
			if w < wmin {
				wmin = w
			}
			if w > wmax {
				wmax = w
			}
		}
	}
	if wmax == 0 {
		wmin, wmax = 0.01, 100
	}
	wmin /= 10
	wmax *= 10

	for i := range 50 {
		w := wmin * math.Pow(wmax/wmin, float64(i)/49)
		freqs = append(freqs, w)
	}

	for _, w := range freqs {
		sv := evalMaxSV(sys, w, p, m)
		if sv > gammaLow {
			gammaLow = sv
			omegaPeak = w
		}
	}

	return gammaLow, omegaPeak
}

func evalMaxSV(sys *System, w float64, p, m int) float64 {
	s := complex(0, w)
	G, err := sys.EvalFr(s)
	if err != nil {
		return 0
	}

	if p == 1 && m == 1 {
		return cmplx.Abs(G[0][0])
	}

	rData := make([]float64, 2*p*m)
	for i := range p {
		for j := range m {
			rData[i*m+j] = real(G[i][j])
			rData[(p+i)*m+j] = imag(G[i][j])
		}
	}

	sv := make([]float64, min(2*p, m))
	wq := make([]float64, 1)
	impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, 2*p, m, rData, m, sv, nil, 1, nil, 1, wq, -1)
	work := make([]float64, int(wq[0]))
	impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, 2*p, m, rData, m, sv, nil, 1, nil, 1, work, len(work))

	if len(sv) == 0 {
		return 0
	}
	return sv[0]
}

func hamiltonianHasImagEigs(sys *System, gamma float64, n, m, p int) bool {
	aRaw := sys.A.RawMatrix()
	bRaw := sys.B.RawMatrix()
	cRaw := sys.C.RawMatrix()
	dRaw := sys.D.RawMatrix()

	dIsZero := allZeroDense(sys.D)
	g2 := gamma * gamma
	nn := 2 * n

	h := make([]float64, nn*nn)

	if dIsZero {
		for i := range n {
			for j := range n {
				h[i*nn+j] = aRaw.Data[i*aRaw.Stride+j]
			}
		}

		bbt := make([]float64, n*n)
		blas64.Gemm(blas.NoTrans, blas.Trans, 1,
			blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
			blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
			0, blas64.General{Rows: n, Cols: n, Stride: n, Data: bbt})
		scale := 1.0 / g2
		for i := range n {
			for j := range n {
				h[i*nn+(n+j)] = scale * bbt[i*n+j]
			}
		}

		ctc := make([]float64, n*n)
		blas64.Gemm(blas.Trans, blas.NoTrans, 1,
			blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
			blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
			0, blas64.General{Rows: n, Cols: n, Stride: n, Data: ctc})
		for i := range n {
			for j := range n {
				h[(n+i)*nn+j] = -ctc[i*n+j]
			}
		}

		for i := range n {
			for j := range n {
				h[(n+i)*nn+(n+j)] = -aRaw.Data[j*aRaw.Stride+i]
			}
		}
	} else {
		r := make([]float64, m*m)
		for i := range m {
			r[i*m+i] = g2
		}
		blas64.Gemm(blas.Trans, blas.NoTrans, -1,
			blas64.General{Rows: p, Cols: m, Stride: dRaw.Stride, Data: dRaw.Data},
			blas64.General{Rows: p, Cols: m, Stride: dRaw.Stride, Data: dRaw.Data},
			1, blas64.General{Rows: m, Cols: m, Stride: m, Data: r})

		if !impl.Dpotrf(blas.Upper, m, r, m) {
			return true
		}

		dtc := make([]float64, m*n)
		blas64.Gemm(blas.Trans, blas.NoTrans, 1,
			blas64.General{Rows: p, Cols: m, Stride: dRaw.Stride, Data: dRaw.Data},
			blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
			0, blas64.General{Rows: m, Cols: n, Stride: n, Data: dtc})
		impl.Dpotrs(blas.Upper, m, n, r, m, dtc, n)

		bt := make([]float64, m*n)
		for i := range n {
			for j := range m {
				bt[j*n+i] = bRaw.Data[i*bRaw.Stride+j]
			}
		}
		rinvBt := make([]float64, m*n)
		copy(rinvBt, bt)
		impl.Dpotrs(blas.Upper, m, n, r, m, rinvBt, n)

		h11 := make([]float64, n*n)
		for i := range n {
			copy(h11[i*n:i*n+n], aRaw.Data[i*aRaw.Stride:i*aRaw.Stride+n])
		}
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
			blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
			blas64.General{Rows: m, Cols: n, Stride: n, Data: dtc},
			1, blas64.General{Rows: n, Cols: n, Stride: n, Data: h11})

		h12 := make([]float64, n*n)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
			blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
			blas64.General{Rows: m, Cols: n, Stride: n, Data: rinvBt},
			0, blas64.General{Rows: n, Cols: n, Stride: n, Data: h12})

		h21 := make([]float64, n*n)
		blas64.Gemm(blas.Trans, blas.NoTrans, -1,
			blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
			blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
			0, blas64.General{Rows: n, Cols: n, Stride: n, Data: h21})

		dRinvDtC := make([]float64, p*n)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
			blas64.General{Rows: p, Cols: m, Stride: dRaw.Stride, Data: dRaw.Data},
			blas64.General{Rows: m, Cols: n, Stride: n, Data: dtc},
			0, blas64.General{Rows: p, Cols: n, Stride: n, Data: dRinvDtC})
		blas64.Gemm(blas.Trans, blas.NoTrans, -1,
			blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
			blas64.General{Rows: p, Cols: n, Stride: n, Data: dRinvDtC},
			1, blas64.General{Rows: n, Cols: n, Stride: n, Data: h21})

		for i := range n {
			for j := range n {
				h[i*nn+j] = h11[i*n+j]
				h[i*nn+(n+j)] = h12[i*n+j]
				h[(n+i)*nn+j] = h21[i*n+j]
				h[(n+i)*nn+(n+j)] = -h11[j*n+i]
			}
		}
	}

	wr := make([]float64, nn)
	wi := make([]float64, nn)
	vs := make([]float64, nn*nn)
	wq := make([]float64, 1)
	impl.Dgees(lapack.SchurHess, lapack.SortNone, nil,
		nn, h, nn, wr, wi, vs, nn, wq, -1, nil)
	work := make([]float64, int(wq[0]))
	_, ok := impl.Dgees(lapack.SchurHess, lapack.SortNone, nil,
		nn, h, nn, wr, wi, vs, nn, work, len(work), nil)
	if !ok {
		return true
	}

	threshold := math.Sqrt(eps()) * gamma
	for i := range nn {
		absLam := math.Sqrt(wr[i]*wr[i] + wi[i]*wi[i])
		if absLam > 0 && math.Abs(wr[i]) < threshold*math.Max(1, absLam/gamma) {
			return true
		}
	}
	return false
}

func allZeroDense(m *mat.Dense) bool {
	if m == nil {
		return true
	}
	raw := m.RawMatrix()
	for i := range raw.Rows {
		for j := range raw.Cols {
			if raw.Data[i*raw.Stride+j] != 0 {
				return false
			}
		}
	}
	return true
}

func frobNormD(D *mat.Dense, p, m int) float64 {
	if D == nil || p == 0 || m == 0 {
		return 0
	}
	return denseNorm(D)
}

func maxSVDense(D *mat.Dense, p, m int) float64 {
	if D == nil || p == 0 || m == 0 {
		return 0
	}
	raw := D.RawMatrix()
	data := make([]float64, p*m)
	for i := range p {
		copy(data[i*m:i*m+m], raw.Data[i*raw.Stride:i*raw.Stride+m])
	}
	sv := make([]float64, min(p, m))
	wq := make([]float64, 1)
	impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, p, m, data, m, sv, nil, 1, nil, 1, wq, -1)
	work := make([]float64, int(wq[0]))
	impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, p, m, data, m, sv, nil, 1, nil, 1, work, len(work))
	if len(sv) == 0 {
		return 0
	}
	return sv[0]
}
