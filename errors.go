package controlsys

import "errors"

var (
	ErrDimensionMismatch = errors.New("controlsys: dimension mismatch")
	ErrSingularTransform = errors.New("controlsys: singular matrix in transformation")
	ErrWrongDomain       = errors.New("controlsys: wrong time domain for operation")
	ErrInvalidSampleTime = errors.New("controlsys: sample time must be positive")
	ErrSingularDenom     = errors.New("controlsys: zero or near-zero leading denominator")
	ErrOverflow          = errors.New("controlsys: coefficient overflow")

	ErrNegativeDelay        = errors.New("controlsys: delay must be non-negative")
	ErrFractionalDelay      = errors.New("controlsys: discrete delay must be non-negative integer")
	ErrNonUniformInputDelay = errors.New("controlsys: AbsorbDelay requires uniform delay per input column")

	ErrZeroInternalDelay = errors.New("controlsys: internal delay must be positive (tau=0 creates algebraic loop)")

	ErrAlgebraicLoop = errors.New("controlsys: algebraic loop: (I-D22) singular")

	ErrDomainMismatch  = errors.New("controlsys: systems must share the same time domain")
	ErrFeedbackDelay   = errors.New("controlsys: feedback with delays not supported")
	ErrMixedDelayTypes = errors.New("controlsys: InternalDelay and IODelay cannot coexist")

	ErrNotSymmetric     = errors.New("controlsys: matrix must be symmetric")
	ErrSchurFailed      = errors.New("controlsys: Schur decomposition failed to converge")
	ErrSingularEquation = errors.New("controlsys: matrix equation is singular or nearly singular")
	ErrNoStabilizing    = errors.New("controlsys: no stabilizing solution exists")
	ErrSingularR        = errors.New("controlsys: R matrix is singular or not positive definite")
	ErrUnstableGramian  = errors.New("controlsys: gramian undefined for unstable system")
	ErrUnstable         = errors.New("controlsys: system is unstable")
	ErrNotMinimal       = errors.New("controlsys: gramian not positive definite; system may not be minimal")
	ErrInvalidOrder     = errors.New("controlsys: reduction order out of range")
	ErrSingularA22      = errors.New("controlsys: A22 block singular; singular perturbation not applicable")
)
