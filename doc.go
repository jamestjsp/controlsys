// Package controlsys provides linear state-space system operations for
// continuous and discrete LTI models. It supports MIMO systems with
// transport delays, state-space and transfer-function representations,
// frequency response analysis, discretization, simulation, and
// model reduction.
//
// Methods that configure delays, names, or notes mutate the receiver,
// so shared systems should be copied with Copy before concurrent
// mutation. Call Validate after direct field edits.
//
// Public model and result structs are mutable by design. Constructors copy
// caller-owned matrix and slice inputs unless the function documentation says
// otherwise. Returned models, matrices, slices, and result structs are owned by
// the caller; mutating them does not update the source model unless the value is
// a receiver or explicitly documented workspace-backed result.
//
// Solver workspaces and simulation buffers are mutable scratch storage. Do not
// share a workspace concurrently. When an option supplies a workspace, returned
// matrices may use that workspace storage and remain valid only until the next
// call that reuses the same workspace.
//
// Nil receivers are unsupported unless a method explicitly documents nil-safe
// behavior.
package controlsys
