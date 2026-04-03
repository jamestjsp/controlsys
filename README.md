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

This package is intended to be usable in production control and estimation code, with the usual caveat that numerical software still needs application-specific validation.

- Pin both `controlsys` and the required gonum fork to explicit versions.
- Validate mission-critical models against an external reference, especially for ill-conditioned realizations and delay-heavy systems.
- `System` values are mutable. Use `Copy` before sharing a model across goroutines that may mutate names, delays, notes, or other receiver state.
- The repository CI runs `go vet ./...` and `go test -v -count=1 -race ./...`; those are the recommended baseline checks for downstream integrations.

## Features

- **Three representations:** state-space, transfer function, zero-pole-gain (ZPK) with bidirectional conversion
- **Frequency response:** Bode, Nyquist, Nichols, singular values
- **Stability analysis:** gain/phase margins, disk margins, bandwidth, damping, root locus
- **Control design:** LQR, LQE (Kalman), LQI, LQG, H₂ synthesis, H∞ synthesis, pole placement, Riccati solvers (CARE/DARE)
- **State estimation:** Extended Kalman Filter (EKF) for nonlinear systems
- **System identification:** Eigensystem Realization Algorithm (ERA) from Markov parameters
- **Nonlinear systems:** Jacobian linearization around operating points; Smith predictor for time-delay plants
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
| `IsStabilizable` | Stabilizability test (unstable modes reachable from input) |
| `IsDetectable` | Detectability test (unstable modes observable from output) |
| `DCGain` | Steady-state (DC) gain |
| `Damp` | Natural frequency, damping ratio, time constant |
| `Margin` | Gain and phase margins (SISO) |
| `AllMargin` | All gain/phase crossover points |
| `DiskMargin` | Disk-based stability margin |
| `Bandwidth` | -3 dB bandwidth |
| `RootLocus` | Root locus as a function of loop gain |

### Control Design

| Function | Description |
|----------|-------------|
| `Lqr` | Continuous-time LQR regulator |
| `Dlqr` | Discrete-time LQR regulator |
| `Lqe` | Kalman filter (observer) gain |
| `Lqi` | LQR with integral action |
| `Lqg` | LQG controller (combined LQR + Kalman filter) |
| `H2Syn` | H₂ optimal controller synthesis from generalized plant |
| `HinfSyn` | H∞ controller synthesis from generalized plant |
| `Place` | Pole placement |
| `Care` | Continuous algebraic Riccati equation |
| `Dare` | Discrete algebraic Riccati equation |
| `SmithPredictor` | Smith predictor for time-delay plants |

### State Estimation

| Function/Type | Description |
|---------------|-------------|
| `NewEKF(model, x0, P0)` | Create an Extended Kalman Filter |
| `(*EKF).Predict(u)` | Propagate state and covariance one step |
| `(*EKF).Update(y)` | Correct state with a measurement |
| `(*EKF).State()` | Return current state estimate |
| `type EKFModel` | Nonlinear model: F, H, Jacobians FJac/HJac, noise Q/R |

### System Identification

| Function/Type | Description |
|---------------|-------------|
| `ERA(markov, order, dt)` | Eigensystem Realization Algorithm — recover state-space model from Markov parameters |
| `type ERAResult` | Result: identified `System`, singular value ratios |

### Nonlinear Systems

| Function/Type | Description |
|---------------|-------------|
| `Linearize(model, x0, u0)` | Jacobian linearization of a nonlinear model around an operating point |
| `type NonlinearModel` | Continuous nonlinear model definition (F, G, H functions) |

### Model Reduction

| Function/Method | Description |
|-----------------|-------------|
| `Reduce` | Staircase reduction (controllability/observability) |
| `MinimalRealization` | Shorthand for full reduction |
| `Balreal` | Balanced realization |
| `Balred` | Balanced truncation / singular perturbation |
| `Ctrb` | Controllability matrix |
| `Obsv` | Observability matrix |
| `CtrbF` | Controllability staircase decomposition |
| `ObsvF` | Observability staircase decomposition |
| `Gram` | Controllability/observability gramian |

### System Norms

| Function | Description |
|----------|-------------|
| `H2Norm` | H2 norm (RMS gain) |
| `HinfNorm` | H-infinity norm (peak gain) |

### Lyapunov Equations

| Function | Description |
|----------|-------------|
| `Lyap(A, Q, opts)` | Solve continuous Lyapunov equation AX + XAᵀ + Q = 0 |
| `DLyap(A, Q, opts)` | Solve discrete Lyapunov equation AXAᵀ − X + Q = 0 |
| `NewLyapunovWorkspace(n)` | Pre-allocate workspace for repeated solves |

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
| `Initial` | Free response to initial state |
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
