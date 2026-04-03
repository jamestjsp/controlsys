package controlsys

import "math/cmplx"

type StabsepResult struct {
	Stable   *System
	Unstable *System
}

func Stabsep(sys *System) (*StabsepResult, error) {
	isStable := func(ev complex128) bool {
		if sys.IsContinuous() {
			return real(ev) < 0
		}
		return cmplx.Abs(ev) < 1
	}

	stable, unstable, err := decomposeByEigenvalues(sys, isStable)
	if err != nil {
		return nil, err
	}

	return &StabsepResult{Stable: stable, Unstable: unstable}, nil
}
