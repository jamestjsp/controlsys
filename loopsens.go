package controlsys

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type LoopsensResult struct {
	So *System
	To *System
	Si *System
	Ti *System
}

func Loopsens(L *System) (*LoopsensResult, error) {
	if L == nil {
		return nil, fmt.Errorf("controlsys: loop transfer function cannot be nil")
	}

	_, m, p := L.Dims()
	if m != p {
		return nil, fmt.Errorf("controlsys: loop transfer must be square (%d inputs, %d outputs): %w", m, p, ErrDimensionMismatch)
	}

	eyeData := make([]float64, m*m)
	for i := 0; i < m; i++ {
		eyeData[i*(m+1)] = 1
	}
	I, err := NewGain(mat.NewDense(m, m, eyeData), L.Dt)
	if err != nil {
		return nil, err
	}

	So, err := Feedback(I, L, -1)
	if err != nil {
		return nil, fmt.Errorf("controlsys: loopsens So computation failed: %w", err)
	}

	To, err := Feedback(L, I, -1)
	if err != nil {
		return nil, fmt.Errorf("controlsys: loopsens To computation failed: %w", err)
	}

	Si, err := Feedback(I, L, -1)
	if err != nil {
		return nil, fmt.Errorf("controlsys: loopsens Si computation failed: %w", err)
	}

	Ti, err := Feedback(L, I, -1)
	if err != nil {
		return nil, fmt.Errorf("controlsys: loopsens Ti computation failed: %w", err)
	}

	return &LoopsensResult{
		So: So,
		To: To,
		Si: Si,
		Ti: Ti,
	}, nil
}
