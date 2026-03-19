# controlsys

Go library for linear state-space control systems.

Supports continuous and discrete LTI models with MIMO capability.

## Install

```bash
go get github.com/jamestjsp/controlsys
```

> **Note:** This package depends on a [gonum fork](https://github.com/jamestjsp/gonum) for additional LAPACK routines. Add this to your `go.mod`:
> ```
> replace gonum.org/v1/gonum => github.com/jamestjsp/gonum v0.17.2-fork
> ```

## Features

- **State-space & transfer function** representations with bidirectional conversion
- **System interconnection:** series, parallel, feedback, append
- **Minimal realization:** controllability/observability staircase reduction
- **Transmission zeros & poles** via staircase decomposition
- **Frequency response:** Bode plots, evaluation at arbitrary complex frequencies
- **Discretization:** bilinear (Tustin) and zero-order hold (ZOH)
- **Simulation:** discrete-time LTI step response
- **Transport delays:** input/output/internal delays, Pade and Thiran approximations, LFT representation

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

### Analysis

| Method | Description |
|--------|-------------|
| `Poles` | Eigenvalues of A |
| `Zeros` | Transmission zeros |
| `IsStable` | Stability check |
| `FreqResponse` | H(jw) at given frequencies |
| `Bode` | Magnitude/phase vs frequency |
| `EvalFr` | Evaluate at arbitrary s |

### Transformation

| Function/Method | Description |
|-----------------|-------------|
| `TransferFunction` | State-space to transfer function |
| `StateSpace` | Transfer function to state-space |
| `Discretize` | Bilinear (Tustin) c2d |
| `DiscretizeZOH` | Zero-order hold c2d |
| `Undiscretize` | Bilinear d2c |
| `Reduce` | Remove uncontrollable/unobservable states |
| `MinimalRealization` | Shorthand for full reduction |
| `AbsorbDelay` | Augment state to absorb discrete transport delays |
| `Pade` | Replace delays with Pade rational approximation |

### Interconnection

| Function | Description |
|----------|-------------|
| `Series` | Cascade connection |
| `Parallel` | Sum connection |
| `Feedback` | Closed-loop with feedback |
| `Append` | Block diagonal concatenation |
| `SafeFeedback` | Feedback with automatic delay handling |

### Simulation

| Method | Description |
|--------|-------------|
| `Simulate` | Discrete-time response to input sequence |

## Core Algorithms

| Algorithm | Purpose |
|-----------|---------|
| Staircase decomposition | Transmission zeros via rank-revealing factorization |
| Column-pivoting QR | Rank determination with incremental condition estimation |
| Row-pivoting RQ | Dual rank-revealing factorization |
| Controllability staircase | Subspace decomposition for reduction and transfer functions |

## License

MIT
