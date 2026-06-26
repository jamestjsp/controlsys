# API and Mutation Audit

This checklist records production-readiness work for the public API surface.
The package is a toolbox-style control-system package, so a broad public API and
a central state-space model are expected. The audit target is consistency, not
shrinking the toolbox shape.

## Method Semantics

Public APIs should make receiver ownership clear:

- Constructors copy caller-owned matrices and slices unless documented otherwise.
- Transformations that return `*System` should leave the receiver unchanged.
- Configuration methods that mutate the receiver should use imperative names such
  as `SetDelay`, `SetInputDelay`, or `SetOutputName`.
- Any method returning shared workspace-backed storage must document that callers
  should copy results they keep across calls.
- Direct field edits remain supported for expert users, but users should call
  `Validate` after edits.

## API Consistency

Review public additions against these rules before release:

- Prefer typed option structs over free-form strings for new APIs.
- Keep nil options as the default behavior for analysis and design routines.
- Return sentinel-wrapped errors for unsupported model families such as
  non-SISO, descriptor, delay, or wrong-time-domain cases.
- Use the project vocabulary from `CONTEXT.md`: model, state-space model,
  feedthrough, sample time, and delay.
- Keep continuous-time and discrete-time behavior explicit in names,
  documentation, or validation errors.

## Release Gates

Before tagging a production release:

- CI must run `go fix ./...`, `go vet ./...`, and `go test -v -count=1 -race ./...`.
- CI must build a downstream module that imports `github.com/jamestjsp/controlsys`
  and applies the documented Gonum fork replacement.
- Public API changes should update `README.md`, `doc.go`, examples, or this audit
  when behavior or ownership expectations change.
- Numerically sensitive changes should include external-reference tests using
  MATLAB or python-control fixtures when possible.

## Current Follow-Up Items

- Audit every exported `*System` method and classify it as mutating,
  non-mutating, or workspace-backed.
- Replace string method selectors with typed constants where that can be done
  without breaking existing users.
- Track required routines from the Gonum fork and upstream them or document why
  each remains fork-only.
