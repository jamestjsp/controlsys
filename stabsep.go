package controlsys

type StabsepResult struct {
	Stable   *System
	Unstable *System
}

func Stabsep(sys *System) (*StabsepResult, error) {
	isStable := func(ev complex128) bool {
		return poleInsideStabilityBoundary(ev, sys.IsContinuous(), 0)
	}

	stable, unstable, err := decomposeByEigenvalues(sys, isStable)
	if err != nil {
		return nil, err
	}

	return &StabsepResult{Stable: stable, Unstable: unstable}, nil
}
