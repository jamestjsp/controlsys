package controlsys

import "math/cmplx"

func poleOnOrOutsideStabilityBoundary(p complex128, continuous bool, tol float64) bool {
	if continuous {
		return real(p) >= -tol
	}
	return cmplx.Abs(p) >= 1-tol
}

func poleStabilityTolerance(p complex128) float64 {
	return 1e-10 * max(1, cmplx.Abs(p))
}
