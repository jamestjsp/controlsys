// Package controlsys provides linear state-space system operations for
// continuous and discrete LTI models. It supports MIMO systems with
// transport delays, state-space and transfer-function representations,
// frequency response analysis, discretization, simulation, and
// model reduction.
//
// Methods that configure delays, names, or notes mutate the receiver,
// so shared systems should be copied with Copy before concurrent
// mutation. Call Validate after direct field edits.
package controlsys
