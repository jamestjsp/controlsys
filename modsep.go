package controlsys

import (
	"fmt"
	"math/cmplx"
)

type ModsepResult struct {
	Slow *System
	Fast *System
}

func Modsep(sys *System, cutoff float64) (*ModsepResult, error) {
	if cutoff <= 0 {
		return nil, fmt.Errorf("controlsys: cutoff must be positive")
	}

	isSlow := func(ev complex128) bool {
		return cmplx.Abs(ev) < cutoff
	}

	slow, fast, err := decomposeByEigenvalues(sys, isSlow)
	if err != nil {
		return nil, err
	}

	return &ModsepResult{Slow: slow, Fast: fast}, nil
}
