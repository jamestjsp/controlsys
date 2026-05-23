package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/mat"
)

type PassivityOptions struct {
	Omega []float64
	Tol   float64
}

type PassivityResult struct {
	Passive          bool
	MinHermitianPart float64
	Frequency        float64
}

func Passive(sys *System, opts *PassivityOptions) (*PassivityResult, error) {
	if sys == nil {
		return nil, fmt.Errorf("Passive: nil system: %w", ErrDimensionMismatch)
	}
	if err := newDescriptorPolicy(sys).requireStandard("Passive"); err != nil {
		return nil, err
	}
	if sys.HasDelay() {
		return nil, fmt.Errorf("Passive: delayed systems are not supported: %w", ErrDescriptorUnsupported)
	}
	stable, err := sys.IsStable()
	if err != nil {
		return nil, err
	}
	if !stable {
		return nil, fmt.Errorf("Passive: %w", ErrUnstable)
	}
	omega, tol := passivityGrid(opts)
	frd, err := sys.FRD(omega)
	if err != nil {
		return nil, err
	}
	return FRDPassive(frd, &PassivityOptions{Tol: tol})
}

func FRDPassive(frd *FRD, opts *PassivityOptions) (*PassivityResult, error) {
	if frd == nil || len(frd.Response) == 0 {
		return nil, fmt.Errorf("FRDPassive: insufficient data: %w", ErrInsufficientData)
	}
	p, m := len(frd.Response[0]), len(frd.Response[0][0])
	if p != m {
		return nil, fmt.Errorf("FRDPassive: model must be square, got %dx%d: %w", p, m, ErrDimensionMismatch)
	}
	_, tol := passivityGrid(opts)
	result := &PassivityResult{Passive: true, MinHermitianPart: math.Inf(1)}
	for k, h := range frd.Response {
		minPart := minHermitianPart(h)
		if minPart < result.MinHermitianPart {
			result.MinHermitianPart = minPart
			if k < len(frd.Omega) {
				result.Frequency = frd.Omega[k]
			}
		}
	}
	result.Passive = result.MinHermitianPart >= -tol
	return result, nil
}

func SpectralFactor(sys *System) (*System, error) {
	if sys == nil {
		return nil, fmt.Errorf("SpectralFactor: nil system: %w", ErrDimensionMismatch)
	}
	if err := newDescriptorPolicy(sys).requireStandard("SpectralFactor"); err != nil {
		return nil, err
	}
	if sys.HasDelay() {
		return nil, fmt.Errorf("SpectralFactor: delayed systems are not supported: %w", ErrDescriptorUnsupported)
	}
	n, m, p := sys.Dims()
	if n != 0 {
		return nil, fmt.Errorf("SpectralFactor: only static gain models are supported by this tracer: %w", ErrDimensionMismatch)
	}
	if m != p {
		return nil, fmt.Errorf("SpectralFactor: model must be square, got %dx%d: %w", p, m, ErrDimensionMismatch)
	}
	D := newDense(p, m)
	for i := 0; i < p; i++ {
		for j := 0; j < m; j++ {
			if i != j && sys.D.At(i, j) != 0 {
				return nil, fmt.Errorf("SpectralFactor: only diagonal positive static gains are supported: %w", ErrDimensionMismatch)
			}
			if i == j {
				v := sys.D.At(i, j)
				if v < 0 {
					return nil, fmt.Errorf("SpectralFactor: negative diagonal value: %w", ErrNotPSD)
				}
				D.Set(i, j, math.Sqrt(v))
			}
		}
	}
	return NewGain(D, sys.Dt)
}

func passivityGrid(opts *PassivityOptions) ([]float64, float64) {
	tol := 1e-9
	if opts != nil && opts.Tol > 0 {
		tol = opts.Tol
	}
	if opts != nil && len(opts.Omega) > 0 {
		return opts.Omega, tol
	}
	return logspace(-2, 2, 120), tol
}

func minHermitianPart(h [][]complex128) float64 {
	if len(h) == 1 && len(h[0]) == 1 {
		return real(h[0][0])
	}
	n := len(h)
	if n == 2 {
		a := real(h[0][0])
		d := real(h[1][1])
		off := (h[0][1] + cmplx.Conj(h[1][0])) / 2
		halfDiff := (a - d) / 2
		radius := math.Sqrt(halfDiff*halfDiff + real(off)*real(off) + imag(off)*imag(off))
		return (a+d)/2 - radius
	}
	dim := 2 * n
	sym := mat.NewSymDense(dim, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			herm := (h[i][j] + cmplx.Conj(h[j][i])) / 2
			re := real(herm)
			im := imag(herm)
			sym.SetSym(i, j, re)
			sym.SetSym(i, n+j, -im)
			sym.SetSym(n+i, j, im)
			sym.SetSym(n+i, n+j, re)
		}
	}
	var eig mat.EigenSym
	if ok := eig.Factorize(sym, false); !ok {
		return math.Inf(-1)
	}
	vals := eig.Values(nil)
	minPart := vals[0]
	for _, v := range vals[1:] {
		if v < minPart {
			minPart = v
		}
	}
	return minPart
}
