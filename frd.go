package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"sort"

	"gonum.org/v1/gonum/lapack"
)

// FRD represents a Frequency Response Data model — measured or computed
// complex response at discrete frequencies.
type FRD struct {
	Response [][][]complex128
	Omega    []float64
	Dt       float64

	InputName  []string
	OutputName []string
}

// NewFRD creates an FRD model from response data and frequency vector.
// response[k] is the p*m complex response matrix at frequency omega[k].
func NewFRD(response [][][]complex128, omega []float64, dt float64) (*FRD, error) {
	if dt < 0 {
		return nil, fmt.Errorf("FRD: negative sample time: %w", ErrInvalidSampleTime)
	}
	if len(response) != len(omega) {
		return nil, fmt.Errorf("FRD: len(response)=%d != len(omega)=%d: %w",
			len(response), len(omega), ErrDimensionMismatch)
	}
	if len(omega) == 0 {
		return &FRD{Dt: dt}, nil
	}

	for i := range omega {
		if omega[i] < 0 {
			return nil, fmt.Errorf("FRD: omega[%d]=%v is negative: %w", i, omega[i], ErrDimensionMismatch)
		}
	}
	if !sort.Float64sAreSorted(omega) {
		return nil, fmt.Errorf("FRD: omega must be sorted ascending: %w", ErrDimensionMismatch)
	}

	if dt > 0 {
		nyquist := math.Pi / dt
		for i, w := range omega {
			if w > nyquist*(1+1e-10) {
				return nil, fmt.Errorf("FRD: omega[%d]=%v exceeds Nyquist frequency %v: %w",
					i, w, nyquist, ErrDimensionMismatch)
			}
		}
	}

	p := len(response[0])
	if p == 0 {
		return nil, fmt.Errorf("FRD: response matrix has zero rows: %w", ErrDimensionMismatch)
	}
	m := len(response[0][0])
	if m == 0 {
		return nil, fmt.Errorf("FRD: response matrix has zero columns: %w", ErrDimensionMismatch)
	}

	for k := range response {
		if len(response[k]) != p {
			return nil, fmt.Errorf("FRD: response[%d] has %d rows, want %d: %w",
				k, len(response[k]), p, ErrDimensionMismatch)
		}
		for i := range response[k] {
			if len(response[k][i]) != m {
				return nil, fmt.Errorf("FRD: response[%d][%d] has %d cols, want %d: %w",
					k, i, len(response[k][i]), m, ErrDimensionMismatch)
			}
		}
	}

	resp, data := newFRDResponseStorage(len(response), p, m)
	copyComplexGridInto(data, response, p, m)
	om := make([]float64, len(omega))
	copy(om, omega)

	return &FRD{
		Response: resp,
		Omega:    om,
		Dt:       dt,
	}, nil
}

// FRD computes the frequency response data model at the given frequencies.
func (sys *System) FRD(omega []float64) (*FRD, error) {
	if len(omega) == 0 {
		return &FRD{Dt: sys.Dt}, nil
	}

	resp, err := sys.FreqResponse(omega)
	if err != nil {
		return nil, err
	}

	_, m, p := sys.Dims()
	nw := len(omega)

	response, data := newFRDResponseStorage(nw, p, m)
	copy(data, resp.Data)

	om := make([]float64, nw)
	copy(om, omega)

	return &FRD{
		Response:   response,
		Omega:      om,
		Dt:         sys.Dt,
		InputName:  copyStringSlice(sys.InputName),
		OutputName: copyStringSlice(sys.OutputName),
	}, nil
}

func newFRDResponseStorage(nw, p, m int) ([][][]complex128, []complex128) {
	response := make([][][]complex128, nw)
	if nw == 0 || p == 0 || m == 0 {
		return response, nil
	}
	rows := make([][]complex128, nw*p)
	data := make([]complex128, nw*p*m)
	for k := 0; k < nw; k++ {
		response[k] = rows[k*p : (k+1)*p]
		block := data[k*p*m : (k+1)*p*m]
		for i := 0; i < p; i++ {
			start := i * m
			response[k][i] = block[start : start+m : start+m]
		}
	}
	return response, data
}

func copyComplexGridInto(dst []complex128, src [][][]complex128, p, m int) {
	pm := p * m
	for k := range src {
		base := k * pm
		for i := 0; i < p; i++ {
			copy(dst[base+i*m:base+(i+1)*m], src[k][i])
		}
	}
}

func copyComplexMatrixInto(dst []complex128, src [][]complex128, rows, cols int) {
	for i := 0; i < rows; i++ {
		copy(dst[i*cols:(i+1)*cols], src[i])
	}
}

func cMulNestedInto(dst []complex128, a, b [][]complex128, ra, ca, cb int) {
	for i := 0; i < ra; i++ {
		row := dst[i*cb : (i+1)*cb]
		for j := 0; j < cb; j++ {
			var sum complex128
			for k := 0; k < ca; k++ {
				sum += a[i][k] * b[k][j]
			}
			row[j] = sum
		}
	}
}

func cAddNestedInto(dst []complex128, a, b [][]complex128, rows, cols int) {
	for i := 0; i < rows; i++ {
		row := dst[i*cols : (i+1)*cols]
		for j := 0; j < cols; j++ {
			row[j] = a[i][j] + b[i][j]
		}
	}
}

func (f *FRD) Dims() (p, m int) {
	if len(f.Response) == 0 {
		return 0, 0
	}
	return len(f.Response[0]), len(f.Response[0][0])
}

func (f *FRD) NumFrequencies() int {
	return len(f.Omega)
}

func (f *FRD) IsContinuous() bool {
	return f.Dt == 0
}

func (f *FRD) IsDiscrete() bool {
	return f.Dt > 0
}

func (f *FRD) At(freqIdx, i, j int) complex128 {
	return f.Response[freqIdx][i][j]
}

// EvalFr returns the p*m complex response at frequency omega[freqIdx].
func (f *FRD) EvalFr(freqIdx int) [][]complex128 {
	p, m := f.Dims()
	result := make([][]complex128, p)
	for i := 0; i < p; i++ {
		result[i] = make([]complex128, m)
		copy(result[i], f.Response[freqIdx][i])
	}
	return result
}

// FreqResponse returns the FRD data as a FreqResponseMatrix for compatibility.
func (f *FRD) FreqResponse() *FreqResponseMatrix {
	p, m := f.Dims()
	nw := len(f.Omega)
	pm := p * m
	data := make([]complex128, nw*pm)
	for k := 0; k < nw; k++ {
		for i := 0; i < p; i++ {
			for j := 0; j < m; j++ {
				data[k*pm+i*m+j] = f.Response[k][i][j]
			}
		}
	}
	return &FreqResponseMatrix{
		Data:       data,
		NFreq:      nw,
		P:          p,
		M:          m,
		InputName:  copyStringSlice(f.InputName),
		OutputName: copyStringSlice(f.OutputName),
	}
}

// Bode computes magnitude (dB) and phase (degrees) from the FRD data.
func (f *FRD) Bode() *BodeResult {
	p, m := f.Dims()
	nw := len(f.Omega)
	pm := p * m
	magDB := make([]float64, nw*pm)
	phase := make([]float64, nw*pm)

	for k := 0; k < nw; k++ {
		for i := 0; i < p; i++ {
			for j := 0; j < m; j++ {
				off := k*pm + i*m + j
				h := f.Response[k][i][j]
				magDB[off] = 20 * math.Log10(cmplx.Abs(h))
				phase[off] = cmplx.Phase(h) * 180 / math.Pi
			}
		}
	}

	for i := 0; i < p; i++ {
		for j := 0; j < m; j++ {
			for k := 1; k < nw; k++ {
				cur := k*pm + i*m + j
				prev := (k-1)*pm + i*m + j
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

	omega := make([]float64, nw)
	copy(omega, f.Omega)

	return &BodeResult{
		Omega:      omega,
		magDB:      magDB,
		phase:      phase,
		p:          p,
		m:          m,
		InputName:  copyStringSlice(f.InputName),
		OutputName: copyStringSlice(f.OutputName),
	}
}

func (f *FRD) Nyquist() (*NyquistResult, error) {
	p, m := f.Dims()
	if p != 1 || m != 1 {
		return nil, fmt.Errorf("FRD Nyquist: SISO only, got %dx%d", p, m)
	}
	nw := len(f.Omega)
	contour := make([]complex128, nw)
	contourN := make([]complex128, nw)
	for k := 0; k < nw; k++ {
		contour[k] = f.Response[k][0][0]
		contourN[k] = cmplx.Conj(contour[k])
	}
	enc := 0
	for k := 1; k < nw; k++ {
		re0, im0 := real(contour[k-1])+1, imag(contour[k-1])
		re1, im1 := real(contour[k])+1, imag(contour[k])
		if (im0 >= 0) != (im1 >= 0) {
			xCross := re0 - im0*(re1-re0)/(im1-im0)
			if xCross < 0 {
				if im1 > im0 {
					enc++
				} else {
					enc--
				}
			}
		}
	}
	omega := make([]float64, nw)
	copy(omega, f.Omega)
	return &NyquistResult{Omega: omega, Contour: contour, ContourN: contourN, Encirclements: enc}, nil
}

func (f *FRD) Sigma() (*SigmaResult, error) {
	p, m := f.Dims()
	nw := len(f.Omega)
	nsv := min(p, m)
	if nsv == 0 {
		omega := make([]float64, nw)
		copy(omega, f.Omega)
		return &SigmaResult{Omega: omega, nSV: 0}, nil
	}
	sv := make([]float64, nw*nsv)
	if p == 1 && m == 1 {
		for k := 0; k < nw; k++ {
			sv[k] = cmplx.Abs(f.Response[k][0][0])
		}
		omega := make([]float64, nw)
		copy(omega, f.Omega)
		return &SigmaResult{Omega: omega, sv: sv, nSV: nsv}, nil
	}

	ws := newFRDSVDWorkspace(p, m)
	for k := 0; k < nw; k++ {
		H := f.Response[k]
		for i := 0; i < p; i++ {
			top := i * ws.cols
			bottom := (p + i) * ws.cols
			for j := 0; j < m; j++ {
				h := H[i][j]
				ws.block[top+j] = real(h)
				ws.block[top+m+j] = -imag(h)
				ws.block[bottom+j] = imag(h)
				ws.block[bottom+m+j] = real(h)
			}
		}
		impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, ws.rows, ws.cols, ws.block, ws.cols,
			ws.fullSV, nil, 1, nil, 1, ws.work, len(ws.work))
		for i := 0; i < nsv; i++ {
			sv[k*nsv+i] = ws.fullSV[2*i]
		}
	}
	omega := make([]float64, nw)
	copy(omega, f.Omega)
	return &SigmaResult{Omega: omega, sv: sv, nSV: nsv}, nil
}

type frdSVDWorkspace struct {
	block  []float64
	fullSV []float64
	work   []float64
	rows   int
	cols   int
}

func newFRDSVDWorkspace(p, m int) *frdSVDWorkspace {
	rows := 2 * p
	cols := 2 * m
	fullSV := make([]float64, min(rows, cols))
	block := make([]float64, rows*cols)
	workQuery := make([]float64, 1)
	impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, rows, cols, block, cols,
		fullSV, nil, 1, nil, 1, workQuery, -1)
	work := make([]float64, int(workQuery[0]))
	return &frdSVDWorkspace{
		block:  block,
		fullSV: fullSV,
		work:   work,
		rows:   rows,
		cols:   cols,
	}
}

func FRDMargin(f *FRD) (*MarginResult, error) {
	p, m := f.Dims()
	if p != 1 || m != 1 {
		return nil, fmt.Errorf("FRDMargin: SISO only, got %dx%d", p, m)
	}
	nw := len(f.Omega)
	if nw < 2 {
		return nil, fmt.Errorf("FRDMargin: need at least 2 frequency points")
	}

	bode := f.Bode()
	magDB := bode.magDB
	phase := bode.phase

	var allGM []float64
	var allGMFreq []float64
	var allPM []float64
	var allPMFreq []float64

	for k := 1; k < nw; k++ {
		ph0, ph1 := phase[k-1], phase[k]

		if (ph0+180)*(ph1+180) <= 0 && math.Abs(ph0-ph1) < 180 {
			frac := math.Abs(ph0+180) / (math.Abs(ph0+180) + math.Abs(ph1+180) + 1e-30)
			magAtCross := magDB[k-1] + frac*(magDB[k]-magDB[k-1])
			gm := -magAtCross
			wCross := bode.Omega[k-1] + frac*(bode.Omega[k]-bode.Omega[k-1])
			allGM = append(allGM, gm)
			allGMFreq = append(allGMFreq, wCross)
		}
	}

	for k := 1; k < nw; k++ {
		m0, m1 := magDB[k-1], magDB[k]

		if m0*m1 <= 0 && math.Abs(m0-m1) < 40 {
			frac := math.Abs(m0) / (math.Abs(m0) + math.Abs(m1) + 1e-30)
			phAtCross := phase[k-1] + frac*(phase[k]-phase[k-1])
			pm := 180 + phAtCross
			wCross := bode.Omega[k-1] + frac*(bode.Omega[k]-bode.Omega[k-1])
			allPM = append(allPM, pm)
			allPMFreq = append(allPMFreq, wCross)
		}
	}

	result := &MarginResult{
		GainMargin:  math.Inf(1),
		PhaseMargin: math.Inf(1),
		WgFreq:      math.NaN(),
		WpFreq:      math.NaN(),
	}

	for i, gm := range allGM {
		if gm > 0 && gm < result.GainMargin {
			result.GainMargin = gm
			result.WpFreq = allGMFreq[i]
		}
	}
	if math.IsInf(result.GainMargin, 1) {
		for i, gm := range allGM {
			if gm > result.GainMargin || math.IsInf(result.GainMargin, 1) {
				result.GainMargin = gm
				result.WpFreq = allGMFreq[i]
			}
		}
	}

	for i, pm := range allPM {
		if pm > 0 && pm < result.PhaseMargin {
			result.PhaseMargin = pm
			result.WgFreq = allPMFreq[i]
		}
	}
	if math.IsInf(result.PhaseMargin, 1) {
		for i, pm := range allPM {
			if pm > result.PhaseMargin || math.IsInf(result.PhaseMargin, 1) {
				result.PhaseMargin = pm
				result.WgFreq = allPMFreq[i]
			}
		}
	}

	return result, nil
}

func frdGridsMatch(f1, f2 *FRD) error {
	if len(f1.Omega) != len(f2.Omega) {
		return fmt.Errorf("frd: frequency grid lengths %d != %d", len(f1.Omega), len(f2.Omega))
	}
	for i := range f1.Omega {
		if math.Abs(f1.Omega[i]-f2.Omega[i]) > 1e-12*math.Max(1, math.Abs(f1.Omega[i])) {
			return fmt.Errorf("frd: frequency mismatch at index %d: %g != %g", i, f1.Omega[i], f2.Omega[i])
		}
	}
	return nil
}

func FRDSeries(f1, f2 *FRD) (*FRD, error) {
	if err := frdGridsMatch(f1, f2); err != nil {
		return nil, err
	}
	p1, m1 := f1.Dims()
	p2, m2 := f2.Dims()
	if p1 != m2 {
		return nil, fmt.Errorf("frd series: f1 outputs %d != f2 inputs %d", p1, m2)
	}

	nw := len(f1.Omega)
	resp, data := newFRDResponseStorage(nw, p2, m1)
	for w := 0; w < nw; w++ {
		cMulNestedInto(data[w*p2*m1:(w+1)*p2*m1], f2.Response[w], f1.Response[w], p2, p1, m1)
	}
	return &FRD{Response: resp, Omega: append([]float64(nil), f1.Omega...), Dt: f1.Dt}, nil
}

func FRDParallel(f1, f2 *FRD) (*FRD, error) {
	if err := frdGridsMatch(f1, f2); err != nil {
		return nil, err
	}
	p1, m1 := f1.Dims()
	p2, m2 := f2.Dims()
	if p1 != p2 || m1 != m2 {
		return nil, fmt.Errorf("frd parallel: dims (%d,%d) != (%d,%d)", p1, m1, p2, m2)
	}

	nw := len(f1.Omega)
	resp, data := newFRDResponseStorage(nw, p1, m1)
	for w := 0; w < nw; w++ {
		cAddNestedInto(data[w*p1*m1:(w+1)*p1*m1], f1.Response[w], f2.Response[w], p1, m1)
	}
	return &FRD{Response: resp, Omega: append([]float64(nil), f1.Omega...), Dt: f1.Dt}, nil
}

func FRDFeedback(plant, controller *FRD, sign float64) (*FRD, error) {
	if err := frdGridsMatch(plant, controller); err != nil {
		return nil, err
	}
	pp, pm := plant.Dims()
	cp, cm := controller.Dims()
	if pp != cm {
		return nil, fmt.Errorf("frd feedback: plant outputs %d != controller inputs %d", pp, cm)
	}
	if pm != cp {
		return nil, fmt.Errorf("frd feedback: plant inputs %d != controller outputs %d", pm, cp)
	}

	nw := len(plant.Omega)
	resp, data := newFRDResponseStorage(nw, pp, pm)
	ws := newFRDFeedbackWorkspace(pp, pm)

	for w := 0; w < nw; w++ {
		copyComplexMatrixInto(ws.g, plant.Response[w], pp, pm)
		copyComplexMatrixInto(ws.k, controller.Response[w], cp, pp)
		cMulInto(ws.kg, ws.k, ws.g, cp, pp, pm)

		for i := range ws.kg {
			ws.ipkg[i] = complex(-sign, 0) * ws.kg[i]
		}
		for i := 0; i < ws.n; i++ {
			ws.ipkg[i*ws.n+i] += 1
		}

		err := cInvertInto(ws.inv, ws.aug, ws.ipkg, ws.n)
		if err != nil {
			return nil, fmt.Errorf("frd feedback: singular at freq index %d", w)
		}
		cMulInto(data[w*pp*pm:(w+1)*pp*pm], ws.g, ws.inv, pp, ws.n, pm)
	}

	return &FRD{Response: resp, Omega: append([]float64(nil), plant.Omega...), Dt: plant.Dt}, nil
}

type frdFeedbackWorkspace struct {
	g    []complex128
	k    []complex128
	kg   []complex128
	ipkg []complex128
	inv  []complex128
	aug  []complex128
	n    int
}

func newFRDFeedbackWorkspace(pp, pm int) *frdFeedbackWorkspace {
	n := pm
	return &frdFeedbackWorkspace{
		g:    make([]complex128, pp*pm),
		k:    make([]complex128, pm*pp),
		kg:   make([]complex128, n*n),
		ipkg: make([]complex128, n*n),
		inv:  make([]complex128, n*n),
		aug:  make([]complex128, n*2*n),
		n:    n,
	}
}
