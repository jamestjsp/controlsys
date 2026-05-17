package controlsys

import "math/cmplx"

type poleStabilityClass int

const (
	poleClassStable poleStabilityClass = iota
	poleClassBoundary
	poleClassUnstable
)

func classifyPoleStability(p complex128, continuous bool, tol float64) poleStabilityClass {
	if continuous {
		switch {
		case real(p) < -tol:
			return poleClassStable
		case real(p) > tol:
			return poleClassUnstable
		default:
			return poleClassBoundary
		}
	}
	r := cmplx.Abs(p)
	switch {
	case r < 1-tol:
		return poleClassStable
	case r > 1+tol:
		return poleClassUnstable
	default:
		return poleClassBoundary
	}
}

func poleOnOrOutsideStabilityBoundary(p complex128, continuous bool, tol float64) bool {
	return classifyPoleStability(p, continuous, tol) != poleClassStable
}

func poleInsideStabilityBoundary(p complex128, continuous bool, tol float64) bool {
	return classifyPoleStability(p, continuous, tol) == poleClassStable
}

func poleOutsideStabilityBoundary(p complex128, continuous bool, tol float64) bool {
	return classifyPoleStability(p, continuous, tol) == poleClassUnstable
}

func poleOnStabilityBoundary(p complex128, continuous bool, tol float64) bool {
	return classifyPoleStability(p, continuous, tol) == poleClassBoundary
}

func poleStabilityTolerance(p complex128) float64 {
	return 1e-10 * max(1, cmplx.Abs(p))
}
