# controlsys

Go library for linear state-space control systems.

Supports continuous and discrete LTI models with MIMO capability.

## Install

```bash
go get github.com/jamestjsp/controlsys
```

> **Note:** This package depends on a [gonum fork](https://github.com/jamestjsp/gonum) for additional LAPACK routines. Because `replace` directives do not propagate to downstream modules, applications that import `controlsys` must add this to their own `go.mod`:
> ```
> replace gonum.org/v1/gonum => github.com/jamestjsp/gonum v0.17.2-fork
> ```

## Production Readiness

This package is suitable for production use when you need numerically robust LTI modeling and analysis in Go. However, it is still a numerical library rather than a turn-key control platform. In practice that means the main production questions are dependency management, numerical validation, and API discipline rather than service-style concerns like logging or network hardening.

### What is already in good shape

- **Input validation:** constructors and most mutating APIs reject dimension mismatches, invalid sample times, negative delays, mixed delay models, and unsupported control-design inputs.
- **Defensive copies at API boundaries:** constructors and delay/name setters copy caller-owned matrices and slices so a system is not accidentally corrupted by later mutations to the original inputs.
- **Verification coverage:** the repository includes focused unit tests, cross-validation tests, MATLAB/MathWorks reference checks, examples, benchmarks, `go vet`, and CI race-test coverage.
- **Operational simplicity:** the dependency surface is intentionally small and the package is pure Go aside from LAPACK-backed numerical routines provided by gonum.

### Known production caveats

- **Forked gonum dependency:** consumers must intentionally pin the gonum fork in their own module. That should be treated as a supply-chain decision and reviewed the same way you would review any vendored numerical runtime.
- **Numerical sensitivity is model-dependent:** ill-conditioned realizations, nearly uncontrollable or nearly unobservable systems, and repeated or clustered poles can produce results that are mathematically valid but sensitive to perturbations. Validate critical models against trusted references.
- **Transport delays are specialized:** continuous delays often require Pade approximation, while discrete fractional delays rely on Thiran-style approximations. Choose the delay model deliberately and verify frequency-domain behavior for the operating range you care about.
- **Mutable systems are not implicitly synchronized:** analysis methods are safe for read-only use, but setters mutate the receiver in place. If a model is shared across goroutines, treat it as immutable or clone it first with `Copy`.
- **Versioning discipline matters:** if you deploy this in a control or estimation stack, pin an explicit module version and rerun your plant- and controller-specific validation on upgrades.

### Recommended release checklist for downstream users

1. Pin both `controlsys` and the gonum fork to explicit versions.
2. Run `go vet ./...` and `go test -v -count=1 -race ./...` in your consuming application.
3. Add model-specific regression tests using non-symmetric `A` matrices so transposition bugs cannot hide behind diagonal fixtures.
4. Benchmark the exact operations that sit on your control loop or planning path with `go test -bench=. -benchmem`.
5. Compare poles, zeros, margins, and representative time responses against an external reference for every mission-critical plant.

## Features

- **Three representations:** state-space, transfer function, zero-pole-gain (ZPK) with bidirectional conversion
- **Frequency response:** Bode, Nyquist, Nichols, singular values
- **Stability analysis:** gain/phase margins, disk margins, bandwidth, damping
- **Control design:** LQR, LQE (Kalman), LQI, pole placement, Riccati solvers (CARE/DARE)
- **Model reduction:** controllability/observability staircase, balanced realization, balanced truncation
- **System norms:** H2 and H-infinity
- **Interconnection:** series, parallel, feedback, append, sum blocks
- **Time-domain:** step, impulse, initial condition, arbitrary input (lsim), discrete simulation
- **Discretization:** ZOH, FOH, Tustin (bilinear), matched pole-zero, discrete-to-discrete resampling
- **Transport delays:** input/output/internal delays, Pade and Thiran approximations, LFT representation
- **Transmission zeros & poles** via staircase decomposition
- **Gramians:** controllability and observability

## Quick Start

```go
package main

import (
	"fmt"

	"github.com/jamestjsp/controlsys"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// Double integrator: x'' = u
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, _ := controlsys.New(A, B, C, D, 0)

	poles, _ := sys.Poles()
	fmt.Println("Poles:", poles)
	stable, _ := sys.IsStable()
	fmt.Println("Stable:", stable)

	tf, _ := sys.TransferFunction(nil)
	fmt.Println("Transfer function:", tf.TF)
}
```

## API Overview

### System Construction

| Function | Description |
|----------|-------------|
| `New` | Create from A, B, C, D matrices |
| `NewWithDelay` | Create with transport delays |
| `NewGain` | Pure feedthrough (D only) |
| `NewFromSlices` | Create from row-major flat arrays |
| `NewZPK` | SISO zero-pole-gain model |
| `NewZPKMIMO` | MIMO zero-pole-gain model |

### Frequency Response & Plotting

| Method | Description |
|--------|-------------|
| `FreqResponse` | H(jw) at given frequencies |
| `Bode` | Magnitude (dB) and phase (deg) vs frequency |
| `Nyquist` | Nyquist plot with encirclement counting |
| `Nichols` | Nichols chart (magnitude vs phase) |
| `Sigma` | Singular value frequency response |
| `EvalFr` | Evaluate at arbitrary complex s |

### Stability & Margins

| Function/Method | Description |
|-----------------|-------------|
| `Poles` | Eigenvalues of A |
| `Zeros` | Transmission zeros |
| `IsStable` | Stability check |
| `DCGain` | Steady-state (DC) gain |
| `Damp` | Natural frequency, damping ratio, time constant |
| `Margin` | Gain and phase margins (SISO) |
| `AllMargin` | All gain/phase crossover points |
| `DiskMargin` | Disk-based stability margin |
| `Bandwidth` | -3 dB bandwidth |

### Control Design

| Function | Description |
|----------|-------------|
| `Lqr` | Continuous-time LQR regulator |
| `Dlqr` | Discrete-time LQR regulator |
| `Lqe` | Kalman filter (observer) gain |
| `Lqi` | LQR with integral action |
| `Place` | Pole placement |
| `Care` | Continuous algebraic Riccati equation |
| `Dare` | Discrete algebraic Riccati equation |

### Model Reduction

| Function/Method | Description |
|-----------------|-------------|
| `Reduce` | Staircase reduction (controllability/observability) |
| `MinimalRealization` | Shorthand for full reduction |
| `Balreal` | Balanced realization |
| `Balred` | Balanced truncation / singular perturbation |
| `Ctrb` | Controllability matrix |
| `Obsv` | Observability matrix |
| `Gram` | Controllability/observability gramian |

### System Norms

| Function | Description |
|----------|-------------|
| `H2Norm` | H2 norm (RMS gain) |
| `HinfNorm` | H-infinity norm (peak gain) |
| `Norm` | General norm computation |

### Representation Conversion

| Function/Method | Description |
|-----------------|-------------|
| `TransferFunction` | State-space → transfer function |
| `StateSpace` | Transfer function → state-space |
| `ZPKModel` | State-space → zero-pole-gain |
| `(ZPK).TransferFunction` | ZPK → transfer function |
| `(TransferFunc).ZPK` | Transfer function → ZPK |

### Discretization

| Method | Description |
|--------|-------------|
| `Discretize` | Bilinear (Tustin) c2d |
| `DiscretizeZOH` | Zero-order hold c2d |
| `DiscretizeFOH` | First-order hold c2d |
| `DiscretizeMatched` | Matched pole-zero c2d |
| `DiscretizeD2D` | Discrete-to-discrete resampling |
| `Undiscretize` | Bilinear d2c |

### Interconnection

| Function | Description |
|----------|-------------|
| `Series` | Cascade connection |
| `Parallel` | Sum connection |
| `Feedback` | Closed-loop with feedback |
| `SafeFeedback` | Feedback with automatic delay handling |
| `Append` | Block diagonal concatenation |
| `Sumblk` | Sum block from string expression |

### Time-Domain Simulation

| Function/Method | Description |
|-----------------|-------------|
| `Step` | Unit step response |
| `Impulse` | Unit impulse response |
| `InitialCondition` | Free response to initial state |
| `Lsim` | Response to arbitrary input signal |
| `Simulate` | Discrete-time simulation |
| `GenSig` | Generate test signals (step, sine, square, pulse) |

### Transport Delays

| Function/Method | Description |
|-----------------|-------------|
| `SetDelay` | Set MIMO delay matrix |
| `SetInputDelay` | Set per-input delays |
| `SetOutputDelay` | Set per-output delays |
| `PadeDelay` | Pade rational approximation |
| `ThiranDelay` | Thiran allpass (fractional discrete delays) |
| `Pade` | Replace all delays with Pade approximations |
| `AbsorbDelay` | Augment state for discrete delays |

## Core Algorithms

| Algorithm | Purpose |
|-----------|---------|
| Staircase decomposition | Transmission zeros via rank-revealing factorization |
| Column-pivoting QR | Rank determination with incremental condition estimation |
| Row-pivoting RQ | Dual rank-revealing factorization |
| Controllability staircase | Subspace decomposition for reduction and transfer functions |
| Balanced realization | Gramian-based state transformation for model reduction |
| Schur decomposition | Riccati equation solvers (CARE/DARE) |

## License

MIT
