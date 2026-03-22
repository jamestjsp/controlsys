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

	aRaw := sys.A.RawMatrix()
	bRaw := sys.B.RawMatrix()
	cRaw := sys.C.RawMatrix()

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

	aData := make([]float64, n*n)
	at := aData[:0:0]
	at = make([]float64, n*n)
	copyStrided(aData, n, aRaw.Data, aRaw.Stride, n, n)
	for i := range n {
		for j := range n {
			at[i*n+j] = aRaw.Data[j*aRaw.Stride+i]
		}
	}

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

	return hsvFromGramians(Wc, Wo, n)
}

func hsvFromGramians(Wc, Wo *mat.Dense, n int) ([]float64, error) {
	wcRaw := Wc.RawMatrix()
	lc := make([]float64, n*n)
	copyStrided(lc, n, wcRaw.Data, wcRaw.Stride, n, n)

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
	copyStrided(lo, n, woRaw.Data, woRaw.Stride, n, n)

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
	wcRaw := Wc.RawMatrix()
	woRaw := Wo.RawMatrix()
	prod := make([]float64, n*n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: n, Cols: n, Stride: wcRaw.Stride, Data: wcRaw.Data},
		blas64.General{Rows: n, Cols: n, Stride: woRaw.Stride, Data: woRaw.Data},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: prod})

	var eig mat.Eigen
	ok := eig.Factorize(mat.NewDense(n, n, prod), mat.EigenNone)
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

	ws := newHamiltonianWS(sys, n, m, p)

	gammaHigh := math.Max(gammaLow*2, 1e-10)
	for range 50 {
		if !ws.hasImagEigs(gammaHigh) {
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
		if ws.hasImagEigs(mid) {
			gammaLow = mid
		} else {
			gammaHigh = mid
		}
	}

	return gammaHigh, omegaPeak, nil
}

// hamiltonianWS holds pre-allocated buffers for the Hamiltonian eigenvalue test
// used across bisection iterations in HinfNorm.
type hamiltonianWS struct {
	n, m, p int
	nn      int
	dIsZero bool

	aData []float64
	aStr  int
	bData []float64
	bStr  int
	cData []float64
	cStr  int
	dData []float64
	dStr  int

	bbt []float64
	ctc []float64

	h    []float64
	wr   []float64
	wi   []float64
	vs   []float64
	work []float64

	r       []float64
	dtc     []float64
	bt      []float64
	rinvBt  []float64
	h11     []float64
	h12     []float64
	h21     []float64
	dRinvDt []float64
}

func newHamiltonianWS(sys *System, n, m, p int) *hamiltonianWS {
	nn := 2 * n
	aRaw := sys.A.RawMatrix()
	bRaw := sys.B.RawMatrix()
	cRaw := sys.C.RawMatrix()
	dRaw := sys.D.RawMatrix()

	ws := &hamiltonianWS{
		n: n, m: m, p: p, nn: nn,
		dIsZero: allZeroDense(sys.D),
		aData:   aRaw.Data, aStr: aRaw.Stride,
		bData:   bRaw.Data, bStr: bRaw.Stride,
		cData:   cRaw.Data, cStr: cRaw.Stride,
		dData:   dRaw.Data, dStr: dRaw.Stride,
		h:       make([]float64, nn*nn),
		wr:      make([]float64, nn),
		wi:      make([]float64, nn),
		vs:      make([]float64, nn*nn),
	}

	ws.bbt = make([]float64, n*n)
	blas64.Gemm(blas.NoTrans, blas.Trans, 1,
		blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
		blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: ws.bbt})

	ws.ctc = make([]float64, n*n)
	blas64.Gemm(blas.Trans, blas.NoTrans, 1,
		blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
		blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: ws.ctc})

	if !ws.dIsZero {
		ws.r = make([]float64, m*m)
		ws.dtc = make([]float64, m*n)
		ws.bt = make([]float64, m*n)
		ws.rinvBt = make([]float64, m*n)
		ws.h11 = make([]float64, n*n)
		ws.h12 = make([]float64, n*n)
		ws.h21 = make([]float64, n*n)
		ws.dRinvDt = make([]float64, p*n)

		for i := range n {
			for j := range m {
				ws.bt[j*n+i] = bRaw.Data[i*bRaw.Stride+j]
			}
		}
	}

	wq := make([]float64, 1)
	impl.Dgees(lapack.SchurHess, lapack.SortNone, nil,
		nn, ws.h, nn, ws.wr, ws.wi, ws.vs, nn, wq, -1, nil)
	ws.work = make([]float64, int(wq[0]))

	return ws
}

func (ws *hamiltonianWS) hasImagEigs(gamma float64) bool {
	n, m, p := ws.n, ws.m, ws.p
	nn := ws.nn
	g2 := gamma * gamma

	h := ws.h
	for i := range len(h) {
		h[i] = 0
	}

	if ws.dIsZero {
		for i := range n {
			for j := range n {
				h[i*nn+j] = ws.aData[i*ws.aStr+j]
			}
		}

		scale := 1.0 / g2
		for i := range n {
			for j := range n {
				h[i*nn+(n+j)] = scale * ws.bbt[i*n+j]
			}
		}

		for i := range n {
			for j := range n {
				h[(n+i)*nn+j] = -ws.ctc[i*n+j]
			}
		}

		for i := range n {
			for j := range n {
				h[(n+i)*nn+(n+j)] = -ws.aData[j*ws.aStr+i]
			}
		}
	} else {
		r := ws.r
		for i := range m * m {
			r[i] = 0
		}
		for i := range m {
			r[i*m+i] = g2
		}
		blas64.Gemm(blas.Trans, blas.NoTrans, -1,
			blas64.General{Rows: p, Cols: m, Stride: ws.dStr, Data: ws.dData},
			blas64.General{Rows: p, Cols: m, Stride: ws.dStr, Data: ws.dData},
			1, blas64.General{Rows: m, Cols: m, Stride: m, Data: r})

		if !impl.Dpotrf(blas.Upper, m, r, m) {
			return true
		}

		dtc := ws.dtc
		blas64.Gemm(blas.Trans, blas.NoTrans, 1,
			blas64.General{Rows: p, Cols: m, Stride: ws.dStr, Data: ws.dData},
			blas64.General{Rows: p, Cols: n, Stride: ws.cStr, Data: ws.cData},
			0, blas64.General{Rows: m, Cols: n, Stride: n, Data: dtc})
		impl.Dpotrs(blas.Upper, m, n, r, m, dtc, n)

		rinvBt := ws.rinvBt
		copy(rinvBt, ws.bt)
		impl.Dpotrs(blas.Upper, m, n, r, m, rinvBt, n)

		h11 := ws.h11
		copyStrided(h11, n, ws.aData, ws.aStr, n, n)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
			blas64.General{Rows: n, Cols: m, Stride: ws.bStr, Data: ws.bData},
			blas64.General{Rows: m, Cols: n, Stride: n, Data: dtc},
			1, blas64.General{Rows: n, Cols: n, Stride: n, Data: h11})

		h12 := ws.h12
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
			blas64.General{Rows: n, Cols: m, Stride: ws.bStr, Data: ws.bData},
			blas64.General{Rows: m, Cols: n, Stride: n, Data: rinvBt},
			0, blas64.General{Rows: n, Cols: n, Stride: n, Data: h12})

		h21 := ws.h21
		blas64.Gemm(blas.Trans, blas.NoTrans, -1,
			blas64.General{Rows: p, Cols: n, Stride: ws.cStr, Data: ws.cData},
			blas64.General{Rows: p, Cols: n, Stride: ws.cStr, Data: ws.cData},
			0, blas64.General{Rows: n, Cols: n, Stride: n, Data: h21})

		dRinvDt := ws.dRinvDt
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
			blas64.General{Rows: p, Cols: m, Stride: ws.dStr, Data: ws.dData},
			blas64.General{Rows: m, Cols: n, Stride: n, Data: dtc},
			0, blas64.General{Rows: p, Cols: n, Stride: n, Data: dRinvDt})
		blas64.Gemm(blas.Trans, blas.NoTrans, -1,
			blas64.General{Rows: p, Cols: n, Stride: ws.cStr, Data: ws.cData},
			blas64.General{Rows: p, Cols: n, Stride: n, Data: dRinvDt},
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

	_, ok := impl.Dgees(lapack.SchurHess, lapack.SortNone, nil,
		nn, h, nn, ws.wr, ws.wi, ws.vs, nn, ws.work, len(ws.work), nil)
	if !ok {
		return true
	}

	threshold := math.Sqrt(eps()) * gamma
	for i := range nn {
		absLam := math.Sqrt(ws.wr[i]*ws.wr[i] + ws.wi[i]*ws.wi[i])
		if absLam > 0 && math.Abs(ws.wr[i]) < threshold*math.Max(1, absLam/gamma) {
			return true
		}
	}
	return false
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
	copyStrided(data, m, raw.Data, raw.Stride, p, m)
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
