package controlsys

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func validateVecResult(context string, v *mat.VecDense, want int) error {
	if v == nil {
		return fmt.Errorf("%s returned nil vector: %w", context, ErrDimensionMismatch)
	}
	if v.Len() != want {
		return fmt.Errorf("%s returned length %d, want %d: %w", context, v.Len(), want, ErrDimensionMismatch)
	}
	return nil
}

func validateDenseResult(context string, m *mat.Dense, wantR, wantC int) error {
	if m == nil {
		return fmt.Errorf("%s returned nil matrix: %w", context, ErrDimensionMismatch)
	}
	r, c := m.Dims()
	if r != wantR || c != wantC {
		return fmt.Errorf("%s returned %dx%d, want %dx%d: %w", context, r, c, wantR, wantC, ErrDimensionMismatch)
	}
	return nil
}
