package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/lapack"
)

type FreqResponseMatrix struct {
	Data       []complex128
	NFreq      int
	P, M       int
	InputName  []string
	OutputName []string
}

func (f *FreqResponseMatrix) At(freq, output, input int) complex128 {
	return f.Data[freq*f.P*f.M+output*f.M+input]
}

type BodeResult struct {
	Omega      []float64
	magDB      []float64
	phase      []float64
	p, m       int
	InputName  []string
	OutputName []string
}

func (b *BodeResult) MagDBAt(freq, output, input int) float64 {
	return b.magDB[freq*b.p*b.m+output*b.m+input]
}

func (b *BodeResult) PhaseAt(freq, output, input int) float64 {
	return b.phase[freq*b.p*b.m+output*b.m+input]
}

func (sys *System) FreqResponse(omega []float64) (*FreqResponseMatrix, error) {
	if len(omega) == 0 {
		return nil, nil
	}

	_, m, p := sys.Dims()

	if sys.HasInternalDelay() {
		resp, err := freqResponseLFT(sys, omega, p, m)
		if err != nil {
			return nil, err
		}
		applyIODelayPhase(sys, omega, resp.Data, p, m)
		resp.InputName = copyStringSlice(sys.InputName)
		resp.OutputName = copyStringSlice(sys.OutputName)
		return resp, nil
	}

	res, err := sys.TransferFunction(nil)
	if err != nil {
		return nil, err
	}

	nw := len(omega)
	pm := p * m
	data := make([]complex128, nw*pm)

	for k, w := range omega {
		var s complex128
		if sys.IsContinuous() {
			s = complex(0, w)
		} else {
			s = cmplx.Exp(complex(0, w*sys.Dt))
		}
		res.TF.evalInto(s, data[k*pm:(k+1)*pm])
	}

	applyIODelayPhase(sys, omega, data, p, m)

	return &FreqResponseMatrix{
		Data: data, NFreq: nw, P: p, M: m,
		InputName:  copyStringSlice(sys.InputName),
		OutputName: copyStringSlice(sys.OutputName),
	}, nil
}

func (sys *System) Bode(omega []float64, nPoints int) (*BodeResult, error) {
	if omega == nil {
		var err2 error
		omega, err2 = autoBodeFreqs(sys, nPoints)
		if err2 != nil {
			return nil, err2
		}
	}

	resp, err := sys.FreqResponse(omega)
	if err != nil {
		return nil, err
	}

	p, m := resp.P, resp.M
	nw := len(omega)
	pm := p * m
	magDB := make([]float64, nw*pm)
	phase := make([]float64, nw*pm)

	for k := range omega {
		for i := 0; i < p; i++ {
			for j := 0; j < m; j++ {
				off := (k*p+i)*m + j
				h := resp.At(k, i, j)
				magDB[off] = 20 * math.Log10(cmplx.Abs(h))
				phase[off] = cmplx.Phase(h) * 180 / math.Pi
			}
		}
	}

	for i := 0; i < p; i++ {
		for j := 0; j < m; j++ {
			for k := 1; k < nw; k++ {
				cur := (k*p+i)*m + j
				prev := ((k-1)*p+i)*m + j
				diff := phase[cur] - phase[prev]
				if diff > 180 {
					phase[cur] -= 360
				}
				if diff < -180 {
					phase[cur] += 360
				}
			}
		}
	}

	return &BodeResult{
		Omega: omega, magDB: magDB, phase: phase, p: p, m: m,
		InputName:  copyStringSlice(sys.InputName),
		OutputName: copyStringSlice(sys.OutputName),
	}, nil
}

func (sys *System) EvalFr(s complex128) ([][]complex128, error) {
	_, m, p := sys.Dims()

	if sys.HasInternalDelay() {
		g, err := evalFrLFT(sys, s, p, m)
		if err != nil {
			return nil, err
		}
		result := make([][]complex128, p)
		for i := 0; i < p; i++ {
			result[i] = make([]complex128, m)
			copy(result[i], g[i*m:(i+1)*m])
		}
		hasIODelay := sys.InputDelay != nil || sys.OutputDelay != nil || sys.Delay != nil
		if hasIODelay {
			for i := 0; i < p; i++ {
				for j := 0; j < m; j++ {
					tau := ioDelayTotal(sys, i, j)
					if tau != 0 {
						if sys.IsContinuous() {
							result[i][j] *= cmplx.Exp(-s * complex(tau, 0))
						} else {
							d := int(math.Round(tau))
							for k := 0; k < d; k++ {
								result[i][j] /= s
							}
						}
					}
				}
			}
		}
		return result, nil
	}

	res, err := sys.TransferFunction(nil)
	if err != nil {
		return nil, err
	}
	vals := res.TF.Eval(s)

	hasIODelay := sys.InputDelay != nil || sys.OutputDelay != nil || sys.Delay != nil
	if hasIODelay {
		for i := 0; i < p; i++ {
			for j := 0; j < m; j++ {
				tau := ioDelayTotal(sys, i, j)
				if tau != 0 {
					if sys.IsContinuous() {
						vals[i][j] *= cmplx.Exp(-s * complex(tau, 0))
					} else {
						d := int(math.Round(tau))
						for k := 0; k < d; k++ {
							vals[i][j] /= s
						}
					}
				}
			}
		}
	}

	return vals, nil
}

func autoBodeFreqs(sys *System, nPoints int) ([]float64, error) {
	if nPoints <= 0 {
		nPoints = 200
	}

	poles, err := sys.Poles()
	if err != nil {
		return nil, err
	}
	var natFreqs []float64
	for _, p := range poles {
		var wn float64
		if sys.IsContinuous() {
			wn = cmplx.Abs(p)
		} else {
			lp := cmplx.Log(p)
			wn = cmplx.Abs(lp) / sys.Dt
		}
		if wn > 0 {
			natFreqs = append(natFreqs, wn)
		}
	}

	wMin, wMax := 0.01, 100.0
	if len(natFreqs) > 0 {
		lo, hi := natFreqs[0], natFreqs[0]
		for _, w := range natFreqs[1:] {
			if w < lo {
				lo = w
			}
			if w > hi {
				hi = w
			}
		}
		wMin = lo / 10
		wMax = hi * 10
		if wMin < 1e-4 {
			wMin = 1e-4
		}
		if wMax > 1e4 {
			wMax = 1e4
		}
	}

	return logspace(math.Log10(wMin), math.Log10(wMax), nPoints), nil
}

func logspace(start, stop float64, n int) []float64 {
	if n == 1 {
		return []float64{math.Pow(10, start)}
	}
	out := make([]float64, n)
	step := (stop - start) / float64(n-1)
	for i := range out {
		out[i] = math.Pow(10, start+float64(i)*step)
	}
	return out
}

func ioDelayTotal(sys *System, i, j int) float64 {
	var tau float64
	if sys.InputDelay != nil {
		tau += sys.InputDelay[j]
	}
	if sys.OutputDelay != nil {
		tau += sys.OutputDelay[i]
	}
	if sys.Delay != nil {
		tau += sys.Delay.At(i, j)
	}
	return tau
}

func applyIODelayPhase(sys *System, omega []float64, data []complex128, p, m int) {
	hasIODelay := sys.InputDelay != nil || sys.OutputDelay != nil || sys.Delay != nil
	if !hasIODelay {
		return
	}
	for k, w := range omega {
		var s complex128
		if sys.IsContinuous() {
			s = complex(0, w)
		} else {
			s = cmplx.Exp(complex(0, w*sys.Dt))
		}
		for i := 0; i < p; i++ {
			for j := 0; j < m; j++ {
				tau := ioDelayTotal(sys, i, j)
				if tau != 0 {
					off := (k*p+i)*m + j
					if sys.IsContinuous() {
						data[off] *= cmplx.Exp(-s * complex(tau, 0))
					} else {
						d := int(math.Round(tau))
						for q := 0; q < d; q++ {
							data[off] /= s
						}
					}
				}
			}
		}
	}
}

func freqResponseLFT(sys *System, omega []float64, p, m int) (*FreqResponseMatrix, error) {
	nw := len(omega)
	data := make([]complex128, nw*p*m)
	n, _, _ := sys.Dims()
	N := len(sys.InternalDelay)
	ws := newLFTWorkspace(n, N, p, m)

	for k, w := range omega {
		var s complex128
		if sys.IsContinuous() {
			s = complex(0, w)
		} else {
			s = cmplx.Exp(complex(0, w*sys.Dt))
		}
		if err := evalFrLFTInto(ws, sys, s, n, N, p, m); err != nil {
			return nil, err
		}
		copy(data[k*p*m:], ws.g[:p*m])
	}

	return &FreqResponseMatrix{Data: data, NFreq: nw, P: p, M: m}, nil
}

type lftWorkspace struct {
	sIA       []complex128
	resolvent []complex128
	hTemp     []complex128
	H11       []complex128
	H12       []complex128
	H21       []complex128
	H22       []complex128
	delta     []complex128
	H22D      []complex128
	ImH22D    []complex128
	invBuf    []complex128
	invResult []complex128
	temp      []complex128
	g         []complex128
}

func newLFTWorkspace(n, N, p, m int) *lftWorkspace {
	maxInv := max(n, N)
	return &lftWorkspace{
		sIA:       make([]complex128, n*n),
		resolvent: make([]complex128, n*n),
		hTemp:     make([]complex128, n*max(m, N, p)),
		H11:       make([]complex128, p*m),
		H12:       make([]complex128, p*N),
		H21:       make([]complex128, N*m),
		H22:       make([]complex128, N*N),
		delta:     make([]complex128, N),
		H22D:      make([]complex128, N*N),
		ImH22D:    make([]complex128, N*N),
		invBuf:    make([]complex128, maxInv*2*maxInv),
		invResult: make([]complex128, N*N),
		temp:      make([]complex128, N*m),
		g:         make([]complex128, p*m),
	}
}

func evalFrLFTInto(ws *lftWorkspace, sys *System, s complex128, n, N, p, m int) error {
	aRaw := sys.A.RawMatrix()
	bRaw := sys.B.RawMatrix()
	cRaw := sys.C.RawMatrix()
	var dData []float64
	var dStride int
	if sys.D != nil {
		dR := sys.D.RawMatrix()
		dData, dStride = dR.Data, dR.Stride
	}

	b2Raw := sys.B2.RawMatrix()
	c2Raw := sys.C2.RawMatrix()
	var d12Data, d21Data, d22Data []float64
	var d12Stride, d21Stride, d22Stride int
	if sys.D12 != nil {
		r := sys.D12.RawMatrix()
		d12Data, d12Stride = r.Data, r.Stride
	}
	if sys.D21 != nil {
		r := sys.D21.RawMatrix()
		d21Data, d21Stride = r.Data, r.Stride
	}
	if sys.D22 != nil {
		r := sys.D22.RawMatrix()
		d22Data, d22Stride = r.Data, r.Stride
	}

	if err := cResolventInto(ws.resolvent, ws.sIA, ws.invBuf, aRaw.Data, aRaw.Stride, s, n); err != nil {
		return err
	}

	cComputeHInto(ws.H11, ws.hTemp, ws.resolvent, cRaw.Data, cRaw.Stride, bRaw.Data, bRaw.Stride, dData, dStride, n, p, m)
	cComputeHInto(ws.H12, ws.hTemp, ws.resolvent, cRaw.Data, cRaw.Stride, b2Raw.Data, b2Raw.Stride, d12Data, d12Stride, n, p, N)
	cComputeHInto(ws.H21, ws.hTemp, ws.resolvent, c2Raw.Data, c2Raw.Stride, bRaw.Data, bRaw.Stride, d21Data, d21Stride, n, N, m)
	cComputeHInto(ws.H22, ws.hTemp, ws.resolvent, c2Raw.Data, c2Raw.Stride, b2Raw.Data, b2Raw.Stride, d22Data, d22Stride, n, N, N)

	cont := sys.IsContinuous()
	for j := 0; j < N; j++ {
		tau := sys.InternalDelay[j]
		if cont {
			ws.delta[j] = cmplx.Exp(-s * complex(tau, 0))
		} else {
			d := int(math.Round(tau))
			ws.delta[j] = 1
			for q := 0; q < d; q++ {
				ws.delta[j] /= s
			}
		}
	}

	cMulDiagRightInto(ws.H22D, ws.H22, ws.delta, N, N)

	for i := 0; i < N*N; i++ {
		ws.ImH22D[i] = -ws.H22D[i]
	}
	for i := 0; i < N; i++ {
		ws.ImH22D[i*N+i] += 1
	}

	if err := cInvertInto(ws.invResult, ws.invBuf, ws.ImH22D, N); err != nil {
		return err
	}

	cMulInto(ws.temp, ws.invResult, ws.H21, N, N, m)
	cMulDiagLeftInto(ws.temp, ws.delta, ws.temp, N, m)

	copy(ws.g[:p*m], ws.H11[:p*m])
	cAddMul(ws.g, ws.H12, ws.temp, p, N, m)

	return nil
}

// evalFrLFT computes G(s) via LFT: G = H11 + H12 * Delta * (I - H22*Delta)^{-1} * H21
// Returns flat p*m complex slice (row-major).
func evalFrLFT(sys *System, s complex128, p, m int) ([]complex128, error) {
	n, _, _ := sys.Dims()
	N := len(sys.InternalDelay)
	ws := newLFTWorkspace(n, N, p, m)
	if err := evalFrLFTInto(ws, sys, s, n, N, p, m); err != nil {
		return nil, err
	}
	result := make([]complex128, p*m)
	copy(result, ws.g[:p*m])
	return result, nil
}

func cResolventInto(dst, sIA, invBuf []complex128, aData []float64, aStride int, s complex128, n int) error {
	if n == 0 {
		return nil
	}
	for i := 0; i < n; i++ {
		row := i * n
		aRow := i * aStride
		for j := 0; j < n; j++ {
			sIA[row+j] = -complex(aData[aRow+j], 0)
		}
		sIA[row+i] += s
	}
	return cInvertInto(dst, invBuf, sIA, n)
}

func cComputeHInto(dst, temp, resolvent []complex128, cData []float64, cStride int, bData []float64, bStride int, dData []float64, dStride, n, rows, cols int) {
	for i := range dst[:rows*cols] {
		dst[i] = 0
	}
	if n > 0 {
		for i := 0; i < n; i++ {
			for j := 0; j < cols; j++ {
				var sum complex128
				for k := 0; k < n; k++ {
					sum += resolvent[i*n+k] * complex(bData[k*bStride+j], 0)
				}
				temp[i*cols+j] = sum
			}
		}
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				var sum complex128
				for k := 0; k < n; k++ {
					sum += complex(cData[i*cStride+k], 0) * temp[k*cols+j]
				}
				dst[i*cols+j] = sum
			}
		}
	}
	if dData != nil {
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				dst[i*cols+j] += complex(dData[i*dStride+j], 0)
			}
		}
	}
}

func cMulInto(dst, a, b []complex128, ar, ac, bc int) {
	for i := 0; i < ar; i++ {
		for j := 0; j < bc; j++ {
			var sum complex128
			for k := 0; k < ac; k++ {
				sum += a[i*ac+k] * b[k*bc+j]
			}
			dst[i*bc+j] = sum
		}
	}
}

func cAddMul(dst, a, b []complex128, ar, ac, bc int) {
	for i := 0; i < ar; i++ {
		for j := 0; j < bc; j++ {
			var sum complex128
			for k := 0; k < ac; k++ {
				sum += a[i*ac+k] * b[k*bc+j]
			}
			dst[i*bc+j] += sum
		}
	}
}

func cMulDiagRightInto(dst, a, diag []complex128, rows, cols int) {
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			dst[i*cols+j] = a[i*cols+j] * diag[j]
		}
	}
}

func cMulDiagLeftInto(dst, diag, a []complex128, rows, cols int) {
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			dst[i*cols+j] = diag[i] * a[i*cols+j]
		}
	}
}

// NicholsResult holds Nichols chart data: open-loop phase (degrees) vs magnitude (dB).
type NicholsResult struct {
	Omega      []float64
	magDB      []float64
	phase      []float64
	p, m       int
	InputName  []string
	OutputName []string
}

func (r *NicholsResult) MagDBAt(freq, output, input int) float64 {
	return r.magDB[freq*r.p*r.m+output*r.m+input]
}

func (r *NicholsResult) PhaseAt(freq, output, input int) float64 {
	return r.phase[freq*r.p*r.m+output*r.m+input]
}

func (sys *System) Nichols(omega []float64, nPoints int) (*NicholsResult, error) {
	bode, err := sys.Bode(omega, nPoints)
	if err != nil {
		return nil, err
	}
	if bode == nil {
		return nil, nil
	}

	phase := bode.phase
	p, m := bode.p, bode.m
	nw := len(bode.Omega)
	pm := p * m

	for i := 0; i < p; i++ {
		for j := 0; j < m; j++ {
			base := phase[i*m+j]
			shift := math.Ceil(base/360) * 360
			for k := 0; k < nw; k++ {
				phase[k*pm+i*m+j] -= shift
			}
		}
	}

	return &NicholsResult{
		Omega:      bode.Omega,
		magDB:      bode.magDB,
		phase:      phase,
		p:          p,
		m:          m,
		InputName:  bode.InputName,
		OutputName: bode.OutputName,
	}, nil
}

// SigmaResult holds singular value frequency response data.
type SigmaResult struct {
	Omega      []float64
	sv         []float64
	nSV        int
	InputName  []string
	OutputName []string
}

func (r *SigmaResult) At(freq, svIndex int) float64 {
	return r.sv[freq*r.nSV+svIndex]
}

func (r *SigmaResult) NSV() int {
	return r.nSV
}

type svdWorkspace struct {
	rData []float64
	sv    []float64
	work  []float64
	p, m  int
}

func newSVDWorkspace(p, m int) *svdWorkspace {
	rows := 2 * p
	cols := m
	nSV := min(rows, cols)
	rData := make([]float64, rows*cols)
	sv := make([]float64, nSV)

	wq := make([]float64, 1)
	impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, rows, cols, rData, cols, sv, nil, 1, nil, 1, wq, -1)
	work := make([]float64, int(wq[0]))

	return &svdWorkspace{rData: rData, sv: sv, work: work, p: p, m: m}
}

func (sys *System) Sigma(omega []float64, nPoints int) (*SigmaResult, error) {
	if omega == nil {
		var err error
		omega, err = autoBodeFreqs(sys, nPoints)
		if err != nil {
			return nil, err
		}
	}

	_, m, p := sys.Dims()
	nSV := min(p, m)
	if nSV == 0 {
		return &SigmaResult{Omega: omega, nSV: 0, InputName: copyStringSlice(sys.InputName), OutputName: copyStringSlice(sys.OutputName)}, nil
	}

	resp, err := sys.FreqResponse(omega)
	if err != nil {
		return nil, err
	}
	if resp == nil {
		return &SigmaResult{Omega: omega, nSV: nSV, InputName: copyStringSlice(sys.InputName), OutputName: copyStringSlice(sys.OutputName)}, nil
	}

	nw := len(omega)
	pm := p * m
	data := resp.Data

	if p == 1 && m == 1 {
		sv := make([]float64, nw)
		for k := range nw {
			sv[k] = cmplx.Abs(data[k])
		}
		return &SigmaResult{Omega: omega, sv: sv, nSV: 1, InputName: copyStringSlice(sys.InputName), OutputName: copyStringSlice(sys.OutputName)}, nil
	}

	ws := newSVDWorkspace(p, m)
	allSV := make([]float64, nw*nSV)
	rData := ws.rData

	for k := range nw {
		base := k * pm
		for i := range p {
			row := i * m
			rowShift := (p + i) * m
			for j := range m {
				h := data[base+row+j]
				rData[row+j] = real(h)
				rData[rowShift+j] = imag(h)
			}
		}
		impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, 2*p, m, rData, m,
			ws.sv, nil, 1, nil, 1, ws.work, len(ws.work))
		copy(allSV[k*nSV:], ws.sv[:nSV])
	}

	return &SigmaResult{Omega: omega, sv: allSV, nSV: nSV, InputName: copyStringSlice(sys.InputName), OutputName: copyStringSlice(sys.OutputName)}, nil
}

func cInvertInto(dst, aug, src []complex128, n int) error {
	if n == 0 {
		return nil
	}
	if n == 1 {
		if src[0] == 0 {
			return fmt.Errorf("controlsys: singular complex matrix: %w", ErrSingularTransform)
		}
		dst[0] = 1 / src[0]
		return nil
	}

	w := 2 * n
	for i := 0; i < n; i++ {
		row := i * w
		copy(aug[row:row+n], src[i*n:(i+1)*n])
		for j := n; j < w; j++ {
			aug[row+j] = 0
		}
		aug[row+n+i] = 1
	}

	maxAbs := 0.0
	for _, v := range src[:n*n] {
		if a := cmplx.Abs(v); a > maxAbs {
			maxAbs = a
		}
	}
	tol := float64(n) * maxAbs * eps()
	if tol == 0 {
		tol = 1e-15
	}

	for col := 0; col < n; col++ {
		pivot := -1
		best := 0.0
		for row := col; row < n; row++ {
			v := cmplx.Abs(aug[row*w+col])
			if v > best {
				best = v
				pivot = row
			}
		}
		if best < tol {
			return fmt.Errorf("controlsys: singular complex matrix: %w", ErrSingularTransform)
		}
		if pivot != col {
			for j := 0; j < w; j++ {
				aug[col*w+j], aug[pivot*w+j] = aug[pivot*w+j], aug[col*w+j]
			}
		}
		inv := 1 / aug[col*w+col]
		for j := col; j < w; j++ {
			aug[col*w+j] *= inv
		}
		for row := 0; row < n; row++ {
			if row == col {
				continue
			}
			factor := aug[row*w+col]
			if factor == 0 {
				continue
			}
			for j := col; j < w; j++ {
				aug[row*w+j] -= factor * aug[col*w+j]
			}
		}
	}

	for i := 0; i < n; i++ {
		copy(dst[i*n:(i+1)*n], aug[i*w+n:i*w+w])
	}
	return nil
}

