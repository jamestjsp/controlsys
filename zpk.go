package controlsys

import (
	"math/cmplx"
	"sort"
)

type ZPK struct {
	Zeros [][][]complex128
	Poles [][][]complex128
	Gain  [][]float64
	Dt    float64
}

func NewZPK(zeros, poles []complex128, gain, dt float64) (*ZPK, error) {
	if dt < 0 {
		return nil, ErrInvalidSampleTime
	}
	if err := validatePoles(zeros); err != nil {
		return nil, err
	}
	if err := validatePoles(poles); err != nil {
		return nil, err
	}
	return &ZPK{
		Zeros: [][][]complex128{{copyComplex(zeros)}},
		Poles: [][][]complex128{{copyComplex(poles)}},
		Gain:  [][]float64{{gain}},
		Dt:    dt,
	}, nil
}

func NewZPKMIMO(zeros, poles [][][]complex128, gain [][]float64, dt float64) (*ZPK, error) {
	if dt < 0 {
		return nil, ErrInvalidSampleTime
	}
	p := len(gain)
	if p == 0 {
		return nil, ErrDimensionMismatch
	}
	m := len(gain[0])
	if m == 0 {
		return nil, ErrDimensionMismatch
	}
	if len(zeros) != p || len(poles) != p {
		return nil, ErrDimensionMismatch
	}
	for i := 0; i < p; i++ {
		if len(gain[i]) != m || len(zeros[i]) != m || len(poles[i]) != m {
			return nil, ErrDimensionMismatch
		}
	}

	zCopy := make([][][]complex128, p)
	pCopy := make([][][]complex128, p)
	gCopy := make([][]float64, p)
	for i := 0; i < p; i++ {
		zCopy[i] = make([][]complex128, m)
		pCopy[i] = make([][]complex128, m)
		gCopy[i] = make([]float64, m)
		copy(gCopy[i], gain[i])
		for j := 0; j < m; j++ {
			if err := validatePoles(zeros[i][j]); err != nil {
				return nil, err
			}
			if err := validatePoles(poles[i][j]); err != nil {
				return nil, err
			}
			zCopy[i][j] = copyComplex(zeros[i][j])
			pCopy[i][j] = copyComplex(poles[i][j])
		}
	}
	return &ZPK{Zeros: zCopy, Poles: pCopy, Gain: gCopy, Dt: dt}, nil
}

func (z *ZPK) Dims() (p, m int) {
	p = len(z.Gain)
	if p > 0 {
		m = len(z.Gain[0])
	}
	return
}

func (z *ZPK) IsContinuous() bool { return z.Dt == 0 }
func (z *ZPK) IsDiscrete() bool   { return z.Dt > 0 }

func (z *ZPK) Eval(s complex128) [][]complex128 {
	p, m := z.Dims()
	result := make([][]complex128, p)
	for i := 0; i < p; i++ {
		result[i] = make([]complex128, m)
		for j := 0; j < m; j++ {
			result[i][j] = zpkEvalChannel(s, z.Zeros[i][j], z.Poles[i][j], z.Gain[i][j])
		}
	}
	return result
}

func zpkEvalChannel(s complex128, zeros, poles []complex128, gain float64) complex128 {
	if gain == 0 {
		return 0
	}
	nz := len(zeros)
	np := len(poles)
	h := complex(gain, 0)

	shared := nz
	if np < shared {
		shared = np
	}
	for k := 0; k < shared; k++ {
		h *= (s - zeros[k]) / (s - poles[k])
	}
	for k := shared; k < nz; k++ {
		h *= s - zeros[k]
	}
	for k := shared; k < np; k++ {
		h /= s - poles[k]
	}
	return h
}

func (z *ZPK) FreqResponse(omega []float64) (*FreqResponseMatrix, error) {
	if len(omega) == 0 {
		return nil, ErrDimensionMismatch
	}
	p, m := z.Dims()
	data := make([]complex128, len(omega)*p*m)
	for k, w := range omega {
		var s complex128
		if z.IsContinuous() {
			s = complex(0, w)
		} else {
			s = cmplx.Exp(complex(0, w*z.Dt))
		}
		off := k * p * m
		for i := 0; i < p; i++ {
			for j := 0; j < m; j++ {
				data[off+i*m+j] = zpkEvalChannel(s, z.Zeros[i][j], z.Poles[i][j], z.Gain[i][j])
			}
		}
	}
	return &FreqResponseMatrix{Data: data, NFreq: len(omega), P: p, M: m}, nil
}

func (z *ZPK) TransferFunction() (*TransferFunc, error) {
	p, m := z.Dims()
	tf := &TransferFunc{
		Num: make([][][]float64, p),
		Den: make([][]float64, p),
		Dt:  z.Dt,
	}

	for i := 0; i < p; i++ {
		commonPoles := commonDenomPoles(z.Poles[i])
		tf.Den[i] = []float64(polyFromComplexRoots(sortConjugatePairs(commonPoles)))
		tf.Num[i] = make([][]float64, m)
		for j := 0; j < m; j++ {
			if z.Gain[i][j] == 0 {
				tf.Num[i][j] = []float64{0}
				continue
			}
			missing := poleSetDifference(commonPoles, z.Poles[i][j])
			zeroPoly := polyFromComplexRoots(sortConjugatePairs(z.Zeros[i][j]))
			if len(missing) > 0 {
				missingPoly := polyFromComplexRoots(sortConjugatePairs(missing))
				zeroPoly = zeroPoly.Mul(missingPoly)
			}
			tf.Num[i][j] = []float64(zeroPoly.Scale(z.Gain[i][j]))
		}
	}
	return tf, nil
}

func (tf *TransferFunc) ZPK() (*ZPK, error) {
	p, m := tf.Dims()
	z := &ZPK{
		Zeros: make([][][]complex128, p),
		Poles: make([][][]complex128, p),
		Gain:  make([][]float64, p),
		Dt:    tf.Dt,
	}

	for i := 0; i < p; i++ {
		rowPoles, err := Poly(tf.Den[i]).Roots()
		if err != nil {
			return nil, err
		}

		z.Zeros[i] = make([][]complex128, m)
		z.Poles[i] = make([][]complex128, m)
		z.Gain[i] = make([]float64, m)
		for j := 0; j < m; j++ {
			z.Poles[i][j] = copyComplex(rowPoles)

			num := tf.Num[i][j]
			if len(num) == 0 || (len(num) == 1 && num[0] == 0) {
				z.Gain[i][j] = 0
				continue
			}

			zeros, err := Poly(num).Roots()
			if err != nil {
				return nil, err
			}
			z.Zeros[i][j] = zeros
			z.Gain[i][j] = num[0] / tf.Den[i][0]
		}
	}
	return z, nil
}

func (sys *System) ZPKModel(opts *TransferFuncOpts) (*ZPKResult, error) {
	tfResult, err := sys.TransferFunction(opts)
	if err != nil {
		return nil, err
	}
	zpk, err := tfResult.TF.ZPK()
	if err != nil {
		return nil, err
	}
	return &ZPKResult{
		ZPK:          zpk,
		MinimalOrder: tfResult.MinimalOrder,
		RowDegrees:   tfResult.RowDegrees,
	}, nil
}

type ZPKResult struct {
	ZPK          *ZPK
	MinimalOrder int
	RowDegrees   []int
}

func (z *ZPK) StateSpace(opts *StateSpaceOpts) (*StateSpaceResult, error) {
	tf, err := z.TransferFunction()
	if err != nil {
		return nil, err
	}
	return tf.StateSpace(opts)
}

func copyComplex(src []complex128) []complex128 {
	if src == nil {
		return nil
	}
	dst := make([]complex128, len(src))
	copy(dst, src)
	return dst
}

func sortConjugatePairs(roots []complex128) []complex128 {
	if len(roots) == 0 {
		return nil
	}
	result := make([]complex128, 0, len(roots))
	var reals []complex128
	var complex_ []complex128
	for _, r := range roots {
		if imag(r) == 0 {
			reals = append(reals, r)
		} else {
			complex_ = append(complex_, r)
		}
	}
	sort.Slice(reals, func(i, j int) bool { return real(reals[i]) < real(reals[j]) })
	result = append(result, reals...)

	used := make([]bool, len(complex_))
	for i := 0; i < len(complex_); i++ {
		if used[i] {
			continue
		}
		r := complex_[i]
		if imag(r) < 0 {
			r = cmplx.Conj(r)
		}
		for k := i + 1; k < len(complex_); k++ {
			if used[k] {
				continue
			}
			tol := 100 * eps() * (1 + cmplx.Abs(r))
			if cmplx.Abs(complex_[k]-cmplx.Conj(r)) < tol {
				used[i] = true
				used[k] = true
				result = append(result, r, cmplx.Conj(r))
				break
			}
		}
		if !used[i] {
			used[i] = true
			result = append(result, r, cmplx.Conj(r))
		}
	}
	return result
}

func commonDenomPoles(channelPoles [][]complex128) []complex128 {
	if len(channelPoles) == 0 {
		return nil
	}
	if len(channelPoles) == 1 {
		return copyComplex(channelPoles[0])
	}

	tol := 100 * eps()
	result := copyComplex(channelPoles[0])
	for j := 1; j < len(channelPoles); j++ {
		avail := copyComplex(result)
		for _, p := range channelPoles[j] {
			found := false
			matchTol := tol * (1 + cmplx.Abs(p))
			for k := 0; k < len(avail); k++ {
				if cmplx.Abs(avail[k]-p) < matchTol {
					avail = append(avail[:k], avail[k+1:]...)
					found = true
					break
				}
			}
			if !found {
				result = append(result, p)
			}
		}
	}
	return result
}

func poleSetDifference(all, subset []complex128) []complex128 {
	tol := 100 * eps()
	remaining := copyComplex(all)
	for _, p := range subset {
		matchTol := tol * (1 + cmplx.Abs(p))
		for k := 0; k < len(remaining); k++ {
			if cmplx.Abs(remaining[k]-p) < matchTol {
				remaining = append(remaining[:k], remaining[k+1:]...)
				break
			}
		}
	}
	return remaining
}

