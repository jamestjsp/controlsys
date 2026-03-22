package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/dsp/fourier"
	"gonum.org/v1/gonum/dsp/window"
	"gonum.org/v1/gonum/mat"
)

type FreqRespEstOpts struct {
	NFFT     int
	Window   func([]float64) []float64
	NOverlap int
	Method   string // "h1" (default), "h2", "fft"
}

type FreqRespEstResult struct {
	H         *FreqResponseMatrix
	Omega     []float64
	Coherence []float64
}

func (r *FreqRespEstResult) CoherenceAt(freq, output, input int) float64 {
	if r.Coherence == nil {
		return 0
	}
	return r.Coherence[freq*r.H.P*r.H.M+output*r.H.M+input]
}

func FreqRespEst(input, output *mat.Dense, dt float64, opts *FreqRespEstOpts) (*FreqRespEstResult, error) {
	if dt <= 0 {
		return nil, ErrInvalidSampleTime
	}
	if input == nil || output == nil {
		return nil, ErrInsufficientData
	}

	m, nIn := input.Dims()
	p, nOut := output.Dims()
	if nIn != nOut {
		return nil, fmt.Errorf("controlsys: input has %d samples, output has %d: %w", nIn, nOut, ErrDimensionMismatch)
	}
	N := nIn
	if N == 0 {
		return nil, ErrInsufficientData
	}

	nfft, winFunc, noverlap, method := resolveOpts(opts, N)

	win := make([]float64, nfft)
	for i := range win {
		win[i] = 1
	}
	winFunc(win)

	winPower := 0.0
	for _, w := range win {
		winPower += w * w
	}
	winPower /= float64(nfft)

	if method == "fft" {
		return freqRespEstFFT(input, output, dt, m, p, N, nfft, win, winPower)
	}

	hop := nfft - noverlap
	nSeg := (N - noverlap) / hop
	if nSeg < 1 {
		nfft = N
		noverlap = 0
		hop = N
		nSeg = 1
		win = make([]float64, nfft)
		for i := range win {
			win[i] = 1
		}
		winFunc(win)
		winPower = 0
		for _, w := range win {
			winPower += w * w
		}
		winPower /= float64(nfft)
	}

	nFreq := nfft/2 + 1
	fft := fourier.NewFFT(nfft)
	seg := make([]float64, nfft)
	coeff := make([]complex128, nFreq)

	inRaw := input.RawMatrix()
	outRaw := output.RawMatrix()

	if m == 1 && p == 1 {
		return welchSISO(fft, seg, coeff, win, inRaw.Data, inRaw.Stride, outRaw.Data, outRaw.Stride,
			dt, nfft, nFreq, hop, nSeg, winPower, method)
	}

	return welchMIMO(fft, seg, coeff, win, input, output,
		dt, m, p, nfft, nFreq, hop, nSeg, winPower, method)
}

func resolveOpts(opts *FreqRespEstOpts, N int) (nfft int, winFunc func([]float64) []float64, noverlap int, method string) {
	nfft = 256
	winFunc = window.Hann
	noverlap = -1
	method = "h1"

	if opts != nil {
		if opts.NFFT > 0 {
			nfft = opts.NFFT
		}
		if opts.Window != nil {
			winFunc = opts.Window
		}
		if opts.NOverlap > 0 {
			noverlap = opts.NOverlap
		}
		if opts.Method != "" {
			method = opts.Method
		}
	}

	if nfft > N {
		nfft = N
	}
	if noverlap < 0 {
		noverlap = nfft / 2
	}
	if noverlap >= nfft {
		noverlap = nfft - 1
	}

	return
}

func welchSISO(fft *fourier.FFT, seg []float64, _ []complex128, win []float64,
	uData []float64, _ int, yData []float64, _ int,
	dt float64, nfft, nFreq, hop, nSeg int, _ float64, method string,
) (*FreqRespEstResult, error) {

	Suu := make([]float64, nFreq)
	Syy := make([]float64, nFreq)
	Syu := make([]complex128, nFreq)

	uCoeff := make([]complex128, nFreq)
	yCoeff := make([]complex128, nFreq)

	for s := 0; s < nSeg; s++ {
		start := s * hop

		for k := 0; k < nfft; k++ {
			seg[k] = uData[start+k] * win[k]
		}
		fft.Coefficients(uCoeff, seg)

		for k := 0; k < nfft; k++ {
			seg[k] = yData[start+k] * win[k]
		}
		fft.Coefficients(yCoeff, seg)

		for f := 0; f < nFreq; f++ {
			Suu[f] += real(uCoeff[f]) * real(uCoeff[f]) + imag(uCoeff[f]) * imag(uCoeff[f])
			Syy[f] += real(yCoeff[f]) * real(yCoeff[f]) + imag(yCoeff[f]) * imag(yCoeff[f])
			Syu[f] += cmplx.Conj(uCoeff[f]) * yCoeff[f]
		}
	}

	invN := 1.0 / float64(nSeg)
	for f := range nFreq {
		Suu[f] *= invN
		Syy[f] *= invN
		Syu[f] *= complex(invN, 0)
	}

	maxSuu := 0.0
	for _, v := range Suu {
		if v > maxSuu {
			maxSuu = v
		}
	}
	sthresh := eps() * maxSuu

	H := make([]complex128, nFreq)
	coh := make([]float64, nFreq)

	for f := range nFreq {
		if Suu[f] < sthresh {
			continue
		}
		if method == "h2" {
			if cmplx.Abs(Syu[f]) < eps()*math.Sqrt(Syy[f]*Suu[f]) {
				continue
			}
			H[f] = complex(Syy[f], 0) / cmplx.Conj(Syu[f])
		} else {
			H[f] = Syu[f] / complex(Suu[f], 0)
		}
		denom := Suu[f] * Syy[f]
		if denom > 0 {
			coh[f] = cmplx.Abs(Syu[f]) * cmplx.Abs(Syu[f]) / denom
		}
	}

	omega := make([]float64, nFreq)
	for f := range nFreq {
		omega[f] = 2 * math.Pi * float64(f) / (float64(nfft) * dt)
	}

	return &FreqRespEstResult{
		H:         &FreqResponseMatrix{Data: H, NFreq: nFreq, P: 1, M: 1},
		Omega:     omega,
		Coherence: coh,
	}, nil
}

func welchMIMO(fft *fourier.FFT, seg []float64, _ []complex128, win []float64,
	input, output *mat.Dense,
	dt float64, m, p, nfft, nFreq, hop, nSeg int, _ float64, _ string,
) (*FreqRespEstResult, error) {
	inR := input.RawMatrix()
	outR := output.RawMatrix()

	// Flat FFT storage: uFlat[ch*nFreq + f], yFlat[ch*nFreq + f]
	uFlat := make([]complex128, m*nFreq)
	yFlat := make([]complex128, p*nFreq)

	Suu := make([]complex128, nFreq*m*m)
	Syu := make([]complex128, nFreq*p*m)
	Syy := make([]complex128, nFreq*p*p)

	mm := m * m
	pm := p * m
	pp := p * p

	for s := range nSeg {
		start := s * hop

		for ch := range m {
			for k := range nfft {
				seg[k] = inR.Data[ch*inR.Stride+start+k] * win[k]
			}
			fft.Coefficients(uFlat[ch*nFreq:(ch+1)*nFreq], seg)
		}
		for ch := range p {
			for k := range nfft {
				seg[k] = outR.Data[ch*outR.Stride+start+k] * win[k]
			}
			fft.Coefficients(yFlat[ch*nFreq:(ch+1)*nFreq], seg)
		}

		// Fused accumulation: iterate channel pairs, then sweep frequencies
		for j1 := range m {
			u1 := uFlat[j1*nFreq : (j1+1)*nFreq]
			for j2 := range m {
				u2 := uFlat[j2*nFreq : (j2+1)*nFreq]
				off := j1*m + j2
				for f := range nFreq {
					Suu[f*mm+off] += cmplx.Conj(u1[f]) * u2[f]
				}
			}
		}
		for i := range p {
			yi := yFlat[i*nFreq : (i+1)*nFreq]
			for j := range m {
				uj := uFlat[j*nFreq : (j+1)*nFreq]
				off := i*m + j
				for f := range nFreq {
					Syu[f*pm+off] += cmplx.Conj(uj[f]) * yi[f]
				}
			}
		}
		for i1 := range p {
			y1 := yFlat[i1*nFreq : (i1+1)*nFreq]
			for i2 := range p {
				y2 := yFlat[i2*nFreq : (i2+1)*nFreq]
				off := i1*p + i2
				for f := range nFreq {
					Syy[f*pp+off] += cmplx.Conj(y1[f]) * y2[f]
				}
			}
		}
	}

	invN := complex(1.0/float64(nSeg), 0)
	for i := range Suu {
		Suu[i] *= invN
	}
	for i := range Syu {
		Syu[i] *= invN
	}
	for i := range Syy {
		Syy[i] *= invN
	}

	H := make([]complex128, nFreq*p*m)
	invBuf := make([]complex128, m*2*m)
	invRes := make([]complex128, m*m)

	for f := range nFreq {
		suu := Suu[f*m*m : (f+1)*m*m]
		syu := Syu[f*p*m : (f+1)*p*m]

		err := cInvertInto(invRes, invBuf, suu, m)
		if err != nil {
			continue
		}

		cMulInto(H[f*p*m:(f+1)*p*m], syu, invRes, p, m, m)
	}

	coh := make([]float64, nFreq*p*m)
	for f := range nFreq {
		for i := range p {
			for j := range m {
				syy_ii := Syy[f*p*p+i*p+i]
				suu_jj := Suu[f*m*m+j*m+j]
				denom := real(syy_ii) * real(suu_jj)
				if denom > 0 {
					s := Syu[f*p*m+i*m+j]
					coh[f*p*m+i*m+j] = (real(s)*real(s) + imag(s)*imag(s)) / denom
				}
			}
		}
	}

	omega := make([]float64, nFreq)
	for f := range nFreq {
		omega[f] = 2 * math.Pi * float64(f) / (float64(nfft) * dt)
	}

	return &FreqRespEstResult{
		H:         &FreqResponseMatrix{Data: H, NFreq: nFreq, P: p, M: m},
		Omega:     omega,
		Coherence: coh,
	}, nil
}

func freqRespEstFFT(input, output *mat.Dense, dt float64, m, p, _, nfft int,
	win []float64, _ float64,
) (*FreqRespEstResult, error) {
	nFreq := nfft/2 + 1
	fft := fourier.NewFFT(nfft)
	seg := make([]float64, nfft)

	inRaw := input.RawMatrix()
	outRaw := output.RawMatrix()

	// Pre-compute all output FFTs once: O(p) instead of O(p*m)
	yCoeffs := make([]complex128, p*nFreq)
	for i := range p {
		for k := range nfft {
			seg[k] = outRaw.Data[i*outRaw.Stride+k] * win[k]
		}
		fft.Coefficients(yCoeffs[i*nFreq:(i+1)*nFreq], seg)
	}

	H := make([]complex128, nFreq*p*m)
	uCoeff := make([]complex128, nFreq)

	for j := range m {
		for k := range nfft {
			seg[k] = inRaw.Data[j*inRaw.Stride+k] * win[k]
		}
		fft.Coefficients(uCoeff, seg)

		for i := range p {
			yC := yCoeffs[i*nFreq : (i+1)*nFreq]
			for f := range nFreq {
				if cmplx.Abs(uCoeff[f]) < 1e-30 {
					continue
				}
				H[f*p*m+i*m+j] = yC[f] / uCoeff[f]
			}
		}
	}

	omega := make([]float64, nFreq)
	for f := range nFreq {
		omega[f] = 2 * math.Pi * float64(f) / (float64(nfft) * dt)
	}

	return &FreqRespEstResult{
		H:     &FreqResponseMatrix{Data: H, NFreq: nFreq, P: p, M: m},
		Omega: omega,
	}, nil
}
