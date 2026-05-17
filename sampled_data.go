package controlsys

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func validateSampledIO(context string, input, output *mat.Dense, dt float64) (m, p, n int, err error) {
	if dt <= 0 {
		return 0, 0, 0, ErrInvalidSampleTime
	}
	if input == nil || output == nil {
		return 0, 0, 0, ErrInsufficientData
	}
	m, nIn := input.Dims()
	p, nOut := output.Dims()
	if nIn != nOut {
		return 0, 0, 0, fmt.Errorf("%s: input has %d samples, output has %d: %w", context, nIn, nOut, ErrDimensionMismatch)
	}
	if nIn == 0 || m == 0 || p == 0 {
		return 0, 0, 0, ErrInsufficientData
	}
	return m, p, nIn, nil
}

func validateMarkovSequence(markov []*mat.Dense, order int, dt float64) (p, m int, err error) {
	if len(markov) == 0 {
		return 0, 0, fmt.Errorf("ERA: empty markov sequence: %w", ErrInsufficientData)
	}
	if order <= 0 {
		return 0, 0, fmt.Errorf("ERA: order must be positive: %w", ErrInvalidOrder)
	}
	if dt <= 0 {
		return 0, 0, fmt.Errorf("ERA: %w", ErrInvalidSampleTime)
	}
	if markov[0] == nil {
		return 0, 0, fmt.Errorf("ERA: markov[0] is nil: %w", ErrInsufficientData)
	}

	p, m = markov[0].Dims()
	if p == 0 || m == 0 {
		return 0, 0, fmt.Errorf("ERA: empty markov[0]: %w", ErrInsufficientData)
	}
	for i := 1; i < len(markov); i++ {
		if markov[i] == nil {
			return 0, 0, fmt.Errorf("ERA: markov[%d] is nil: %w", i, ErrInsufficientData)
		}
		ri, ci := markov[i].Dims()
		if ri != p || ci != m {
			return 0, 0, fmt.Errorf("ERA: markov[%d] is %dx%d, expected %dx%d: %w",
				i, ri, ci, p, m, ErrDimensionMismatch)
		}
	}

	minLen := 2*order + 1
	if len(markov) < minLen {
		return 0, 0, fmt.Errorf("ERA: need >= %d markov params for order %d, got %d: %w",
			minLen, order, len(markov), ErrInsufficientData)
	}
	return p, m, nil
}
