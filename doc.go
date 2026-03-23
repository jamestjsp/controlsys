// Package controlsys provides linear state-space system operations for
// continuous and discrete LTI models. It supports MIMO systems with
// transport delays, state-space and transfer-function representations,
// frequency response analysis, discretization, simulation, and
// model reduction.
//
// Constructors defensively copy caller-owned matrices and slices so a
// System can be treated as a stable value after creation. Methods that
// configure delays, names, or notes mutate the receiver, so shared
// systems should be copied with Copy before concurrent mutation.
package controlsys
