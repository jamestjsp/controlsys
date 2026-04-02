package controlsys

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// ERAResult holds the identified state-space model and the Hankel
// singular values from the Eigensystem Realization Algorithm.
type ERAResult struct {
	Sys *System
	HSV []float64
}

// ERA performs the Eigensystem Realization Algorithm to identify a
// discrete-time state-space model from Markov parameters (impulse
// response matrices). Each markov[k] is a p×m matrix representing
// Y(k). The order parameter specifies the desired state dimension
// and dt is the sample time.
func ERA(markov []*mat.Dense, order int, dt float64) (*ERAResult, error) {
	if len(markov) == 0 {
		return nil, fmt.Errorf("ERA: empty markov sequence: %w", ErrInsufficientData)
	}
	if order <= 0 {
		return nil, fmt.Errorf("ERA: order must be positive: %w", ErrInvalidOrder)
	}
	if dt <= 0 {
		return nil, fmt.Errorf("ERA: %w", ErrInvalidSampleTime)
	}

	p, m := markov[0].Dims()
	for i := 1; i < len(markov); i++ {
		ri, ci := markov[i].Dims()
		if ri != p || ci != m {
			return nil, fmt.Errorf("ERA: markov[%d] is %d×%d, expected %d×%d: %w",
				i, ri, ci, p, m, ErrDimensionMismatch)
		}
	}

	minLen := 2*order + 1
	if len(markov) < minLen {
		return nil, fmt.Errorf("ERA: need >= %d markov params for order %d, got %d: %w",
			minLen, order, len(markov), ErrInsufficientData)
	}

	r := len(markov) / 2

	h0Rows, h0Cols := r*p, r*m
	h0Data := make([]float64, h0Rows*h0Cols)
	h1Data := make([]float64, h0Rows*h0Cols)

	for i := 0; i < r; i++ {
		for j := 0; j < r; j++ {
			idx0 := i + j + 1
			idx1 := i + j + 2
			mk0 := markov[idx0].RawMatrix()
			for bi := 0; bi < p; bi++ {
				srcOff := bi * mk0.Stride
				dstOff := (i*p+bi)*h0Cols + j*m
				copy(h0Data[dstOff:dstOff+m], mk0.Data[srcOff:srcOff+m])
			}
			if idx1 < len(markov) {
				mk1 := markov[idx1].RawMatrix()
				for bi := 0; bi < p; bi++ {
					srcOff := bi * mk1.Stride
					dstOff := (i*p+bi)*h0Cols + j*m
					copy(h1Data[dstOff:dstOff+m], mk1.Data[srcOff:srcOff+m])
				}
			}
		}
	}

	h0 := mat.NewDense(h0Rows, h0Cols, h0Data)
	h1 := mat.NewDense(h0Rows, h0Cols, h1Data)

	var svd mat.SVD
	if !svd.Factorize(h0, mat.SVDFull) {
		return nil, fmt.Errorf("ERA: SVD failed to converge")
	}

	allSigma := svd.Values(nil)
	hsv := make([]float64, len(allSigma))
	copy(hsv, allSigma)

	if order > len(allSigma) {
		return nil, fmt.Errorf("ERA: order %d exceeds rank %d: %w",
			order, len(allSigma), ErrInvalidOrder)
	}

	var uFull, vFull mat.Dense
	svd.UTo(&uFull)
	svd.VTo(&vFull)

	un := extractBlock(&uFull, 0, 0, h0Rows, order)
	vn := extractBlock(&vFull, 0, 0, h0Cols, order)

	snHalfInvData := make([]float64, order)
	snHalfData := make([]float64, order)
	for i := 0; i < order; i++ {
		s := allSigma[i]
		sq := math.Sqrt(s)
		snHalfData[i] = sq
		snHalfInvData[i] = 1.0 / sq
	}

	// A = Sn^{-1/2} * Un' * H1 * Vn * Sn^{-1/2}
	var utH1 mat.Dense
	utH1.Mul(un.T(), h1)
	var utH1V mat.Dense
	utH1V.Mul(&utH1, vn)

	aData := make([]float64, order*order)
	utH1VRaw := utH1V.RawMatrix()
	for i := 0; i < order; i++ {
		for j := 0; j < order; j++ {
			aData[i*order+j] = snHalfInvData[i] * utH1VRaw.Data[i*utH1VRaw.Stride+j] * snHalfInvData[j]
		}
	}
	A := mat.NewDense(order, order, aData)

	// B = first m columns of Sn^{1/2} * Vn'
	bData := make([]float64, order*m)
	vnRaw := vn.RawMatrix()
	for i := 0; i < order; i++ {
		for j := 0; j < m; j++ {
			bData[i*m+j] = snHalfData[i] * vnRaw.Data[j*vnRaw.Stride+i]
		}
	}
	B := mat.NewDense(order, m, bData)

	// C = first p rows of Un * Sn^{1/2}
	cData := make([]float64, p*order)
	unRaw := un.RawMatrix()
	for i := 0; i < p; i++ {
		for j := 0; j < order; j++ {
			cData[i*order+j] = unRaw.Data[i*unRaw.Stride+j] * snHalfData[j]
		}
	}
	C := mat.NewDense(p, order, cData)

	D := mat.DenseCopyOf(markov[0])

	sys, err := New(A, B, C, D, dt)
	if err != nil {
		return nil, fmt.Errorf("ERA: %w", err)
	}

	return &ERAResult{Sys: sys, HSV: hsv}, nil
}
