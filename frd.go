package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"sort"

	"gonum.org/v1/gonum/mat"
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

	resp := make([][][]complex128, len(response))
	for k := range response {
		resp[k] = make([][]complex128, p)
		for i := range response[k] {
			resp[k][i] = make([]complex128, m)
			copy(resp[k][i], response[k][i])
		}
	}
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

	response := make([][][]complex128, nw)
	for k := 0; k < nw; k++ {
		response[k] = make([][]complex128, p)
		for i := 0; i < p; i++ {
			response[k][i] = make([]complex128, m)
			for j := 0; j < m; j++ {
				response[k][i][j] = resp.At(k, i, j)
			}
		}
	}

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
	nsv := p
	if m < p {
		nsv = m
	}
	sv := make([]float64, nw*nsv)
	for k := 0; k < nw; k++ {
		H := f.Response[k]
		svd := new(mat.SVD)
		hMat := mat.NewDense(p, m, nil)
		for i := 0; i < p; i++ {
			for j := 0; j < m; j++ {
				hMat.Set(i, j, cmplx.Abs(H[i][j]))
			}
		}
		if p == 1 && m == 1 {
			sv[k] = cmplx.Abs(H[0][0])
			continue
		}
		hReal := mat.NewDense(2*p, 2*m, nil)
		for i := 0; i < p; i++ {
			for j := 0; j < m; j++ {
				hReal.Set(i, j, real(H[i][j]))
				hReal.Set(i, m+j, -imag(H[i][j]))
				hReal.Set(p+i, j, imag(H[i][j]))
				hReal.Set(p+i, m+j, real(H[i][j]))
			}
		}
		if !svd.Factorize(hReal, mat.SVDNone) {
			return nil, fmt.Errorf("FRD Sigma: SVD failed at freq index %d", k)
		}
		vals := svd.Values(nil)
		for s := 0; s < nsv; s++ {
			sv[k*nsv+s] = vals[s]
		}
	}
	omega := make([]float64, nw)
	copy(omega, f.Omega)
	return &SigmaResult{Omega: omega, sv: sv, nSV: nsv}, nil
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

	var gmDB, pm, wcg, wcp float64
	gmDB = math.Inf(1)
	foundGM, foundPM := false, false

	for k := 1; k < nw; k++ {
		h0 := f.Response[k-1][0][0]
		h1 := f.Response[k][0][0]
		ph0 := cmplx.Phase(h0) * 180 / math.Pi
		ph1 := cmplx.Phase(h1) * 180 / math.Pi
		mag0 := cmplx.Abs(h0)
		mag1 := cmplx.Abs(h1)
		magDB0 := 20 * math.Log10(mag0)
		magDB1 := 20 * math.Log10(mag1)

		if (ph0+180)*(ph1+180) <= 0 || (math.Abs(ph0+180) < 1 && math.Abs(ph1+180) < 1) {
			t := math.Abs(ph0+180) / (math.Abs(ph0+180) + math.Abs(ph1+180) + 1e-30)
			magAtCross := magDB0 + t*(magDB1-magDB0)
			gm := -magAtCross
			if !foundGM || gm < gmDB {
				gmDB = gm
				wcg = f.Omega[k-1] + t*(f.Omega[k]-f.Omega[k-1])
				foundGM = true
			}
		}

		if (magDB0)*(magDB1) <= 0 || (math.Abs(magDB0) < 0.5 && math.Abs(magDB1) < 0.5) {
			t := math.Abs(magDB0) / (math.Abs(magDB0) + math.Abs(magDB1) + 1e-30)
			phAtCross := ph0 + t*(ph1-ph0)
			pmCand := 180 + phAtCross
			if !foundPM || f.Omega[k-1]+t*(f.Omega[k]-f.Omega[k-1]) < wcp {
				pm = pmCand
				wcp = f.Omega[k-1] + t*(f.Omega[k]-f.Omega[k-1])
				foundPM = true
			}
		}
	}

	if !foundGM {
		gmDB = math.Inf(1)
	}
	if !foundPM {
		pm = math.Inf(1)
	}

	return &MarginResult{
		GainMargin:  gmDB,
		PhaseMargin: pm,
		WgFreq:      wcp,
		WpFreq:      wcg,
	}, nil
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
	resp := make([][][]complex128, nw)
	for w := 0; w < nw; w++ {
		resp[w] = cMatMul(f2.Response[w], f1.Response[w], p2, p1, m1)
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
	resp := make([][][]complex128, nw)
	for w := 0; w < nw; w++ {
		r := make([][]complex128, p1)
		for i := 0; i < p1; i++ {
			r[i] = make([]complex128, m1)
			for j := 0; j < m1; j++ {
				r[i][j] = f1.Response[w][i][j] + f2.Response[w][i][j]
			}
		}
		resp[w] = r
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
	resp := make([][][]complex128, nw)

	for w := 0; w < nw; w++ {
		G := plant.Response[w]
		K := controller.Response[w]
		KG := cMatMul(K, G, cp, pp, pm)

		n := pm
		IpKG := make([][]complex128, n)
		for i := 0; i < n; i++ {
			IpKG[i] = make([]complex128, n)
			for j := 0; j < n; j++ {
				IpKG[i][j] = complex(-sign, 0) * KG[i][j]
				if i == j {
					IpKG[i][j] += 1
				}
			}
		}

		inv, err := cMatInv(IpKG, n)
		if err != nil {
			return nil, fmt.Errorf("frd feedback: singular at freq index %d", w)
		}
		resp[w] = cMatMul(G, inv, pp, n, pm)
	}

	return &FRD{Response: resp, Omega: append([]float64(nil), plant.Omega...), Dt: plant.Dt}, nil
}

func cMatMul(a, b [][]complex128, ra, ca, cb int) [][]complex128 {
	r := make([][]complex128, ra)
	for i := 0; i < ra; i++ {
		r[i] = make([]complex128, cb)
		for j := 0; j < cb; j++ {
			var s complex128
			for k := 0; k < ca; k++ {
				s += a[i][k] * b[k][j]
			}
			r[i][j] = s
		}
	}
	return r
}

func cMatInv(a [][]complex128, n int) ([][]complex128, error) {
	aug := make([][]complex128, n)
	for i := 0; i < n; i++ {
		aug[i] = make([]complex128, 2*n)
		copy(aug[i][:n], a[i])
		aug[i][n+i] = 1
	}
	for col := 0; col < n; col++ {
		pivot := -1
		var best float64
		for row := col; row < n; row++ {
			if v := cmplx.Abs(aug[row][col]); v > best {
				best = v
				pivot = row
			}
		}
		if best < 1e-15 {
			return nil, fmt.Errorf("singular")
		}
		aug[col], aug[pivot] = aug[pivot], aug[col]
		s := 1.0 / aug[col][col]
		for j := col; j < 2*n; j++ {
			aug[col][j] *= s
		}
		for row := 0; row < n; row++ {
			if row == col {
				continue
			}
			f := aug[row][col]
			for j := col; j < 2*n; j++ {
				aug[row][j] -= f * aug[col][j]
			}
		}
	}
	inv := make([][]complex128, n)
	for i := 0; i < n; i++ {
		inv[i] = make([]complex128, n)
		copy(inv[i], aug[i][n:])
	}
	return inv, nil
}
