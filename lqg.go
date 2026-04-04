package controlsys

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type LqgResult struct {
	Controller *System
	K          *mat.Dense
	L          *mat.Dense
	Xc         *mat.Dense
	Xf         *mat.Dense
}

// Lqg computes the linear-quadratic-Gaussian regulator for a state-space system.
// Q, R are state/input weights for LQR; Qn, Rn are process/measurement noise covariances.
// Returns the observer-based controller and all intermediate gains and Riccati solutions.
func Lqg(sys *System, Q, R, Qn, Rn *mat.Dense, opts *RiccatiOpts) (*LqgResult, error) {
	n, _, _ := sys.Dims()
	if n == 0 {
		return nil, fmt.Errorf("Lqg: system has no states: %w", ErrDimensionMismatch)
	}
	if sys.IsDescriptor() {
		return nil, fmt.Errorf("Lqg: %w", ErrDescriptorRiccati)
	}

	var kRes *RiccatiResult
	var err error
	if sys.IsContinuous() {
		kRes, err = Lqr(sys.A, sys.B, Q, R, opts)
	} else {
		kRes, err = Dlqr(sys.A, sys.B, Q, R, opts)
	}
	if err != nil {
		return nil, fmt.Errorf("Lqg: %w", err)
	}

	lRes, err := Kalman(sys, Qn, Rn, opts)
	if err != nil {
		return nil, fmt.Errorf("Lqg: %w", err)
	}

	ctrl, err := Reg(sys, kRes.K, lRes.K)
	if err != nil {
		return nil, fmt.Errorf("Lqg: %w", err)
	}

	return &LqgResult{
		Controller: ctrl,
		K:          kRes.K,
		L:          lRes.K,
		Xc:        kRes.X,
		Xf:        lRes.X,
	}, nil
}
