# python-control Numerical Audit for controlsys Hardening

Compiled from 100+ GitHub issues, 40+ PRs, discussions, test suites, and
14 Jupyter notebook examples from
[python-control/python-control](https://github.com/python-control/python-control).

Focus: numerical discrepancies vs MATLAB, edge cases, and regression patterns
that apply to our Go implementation.

---

## Table of Contents

1. [Riccati Solvers (CARE/DARE)](#1-riccati-solvers-caredare)
2. [LQR / LQE / Kalman](#2-lqr--lqe--kalman)
3. [Lyapunov Solvers](#3-lyapunov-solvers)
4. [Pole Placement](#4-pole-placement)
5. [Controllability & Observability](#5-controllability--observability)
6. [Gramians & Hankel Singular Values](#6-gramians--hankel-singular-values)
7. [Canonical Forms & Modal Decomposition](#7-canonical-forms--modal-decomposition)
8. [Transfer Function / State Space Conversion](#8-transfer-function--state-space-conversion)
9. [System Interconnection (Feedback, Series, Parallel)](#9-system-interconnection)
10. [Stability Margins](#10-stability-margins)
11. [Frequency Response & Bode Plots](#11-frequency-response--bode-plots)
12. [Time-Domain Simulation](#12-time-domain-simulation)
13. [Discretization (c2d / d2c)](#13-discretization-c2d--d2c)
14. [Model Reduction (Balanced Realization)](#14-model-reduction)
15. [System Norms (H2, H-infinity)](#15-system-norms-h2-h-infinity)
16. [DC Gain & Zeros](#16-dc-gain--zeros)
17. [Nyquist Plots](#17-nyquist-plots)
18. [Numerical Primitives & General Patterns](#18-numerical-primitives--general-patterns)

---

## 1. Riccati Solvers (CARE/DARE)

### 1.1 Symmetry Check Must Be Scale-Aware

**Source**: Issue #1174, PR #348

The elementwise check `(M - M.T) < eps` is broken in two ways:
- Missing `abs()`: small negative roundoff passes the check incorrectly
- Fixed `eps` doesn't scale with matrix magnitude

**Correct approach** (from scipy):
```
norm(M - M.T, 1) > spacing(norm(M, 1)) * 100
```

**Test case**:
```go
// Matrix that is symmetric up to roundoff
Q := mat.NewDense(2, 2, []float64{1.0, 0.5 + 1e-16, 0.5 - 1e-16, 1.0})
// Should pass symmetry check
```

### 1.2 DARE Wrong Results vs MATLAB

**Source**: Issue #8, #157

scipy's `solve_discrete_are` uses van Dooren's generalized eigenvalue method;
MATLAB uses Arnold-Laub (SLICOT sb02od) which is more robust for
ill-conditioned A matrices.

**Test case** (ill-conditioned):
```go
// System where scipy fails but MATLAB succeeds
// A with condition number > 1e8
A := mat.NewDense(2, 2, []float64{1e4, 1, 0, 1e-4})
B := mat.NewDense(2, 1, []float64{1, 1})
Q := eye(2)
R := eye(1)
// Should produce stabilizing solution, not error
```

### 1.3 Generalized DARE with S != 0, E != I

**Source**: Issue #36 (open 8+ years), python-control test suite

scipy only handles S=0, E=I. MATLAB handles the full generalized form.

**Test matrices** (from python-control mateqn_test.py):
```go
A := []float64{-2, -1, -1, -1}   // 2x2
Q := []float64{0, 0, 0, 1}
B := []float64{1, 0, 0, 4}
R := []float64{2, 0, 0, 1}
S := []float64{0, 0, 0, 0}
E := []float64{2, 1, 1, 2}
// Verify: A'XE + E'XA - (E'XB+S)*inv(R)*(B'XE+S') + Q = 0
```

### 1.4 Q/R Positive Definiteness Validation

**Source**: Issue #252, #274

LQR/DARE should validate that the combined cost `[Q N; N' R]` is positive
semi-definite before solving. Non-PD costs produce eigenvalues with real parts
~1e-16 that flip sign randomly.

**Test case**:
```go
// Non-PD cost should error, not silently produce garbage
Q := mat.NewDense(2, 2, []float64{1, 0, 0, 0})
R := mat.NewDense(1, 1, []float64{-1})  // negative R
// Lqr(A, B, Q, R) should return error
```

### 1.5 CARE/DARE Verification Pattern

Always verify the residual equation:
- **CARE**: `A'X + XA - XBR^{-1}B'X + Q = 0`
- **DARE**: `X = A'XA - A'XB(B'XB+R)^{-1}B'XA + Q`
- **Always** check closed-loop stability: eigenvalues in LHP (continuous) or
  inside unit circle (discrete)

---

## 2. LQR / LQE / Kalman

### 2.1 dlqe: Filter Gain vs Predictor Gain

**Source**: Issue #1173 (open)

python-control's `dlqe` returns the predictor gain:
```
L_pred = A*P*C'*(C*P*C'+R)^{-1}
```
MATLAB returns the filter gain:
```
L_filt = P*C'*(C*P*C'+R)^{-1}
```

When A is singular, you **cannot** recover filter gain from predictor gain.

**Test case**:
```go
// Singular A -- filter vs predictor gain differs fundamentally
A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})  // double integrator
B := mat.NewDense(2, 1, []float64{0, 1})
C := mat.NewDense(1, 2, []float64{1, 0})
Qn := eye(2)
Rn := eye(1)
// L_filt = P*C'*inv(C*P*C'+Rn)
// L_pred = A*L_filt  -- but A is singular!
```

### 2.2 lqe Default Noise Input Matrix

**Source**: Issue #1159 (open)

When calling `lqe(sys, QN, RN)` without explicit G (disturbance input matrix),
the default differs between MATLAB/Octave and python-control. Dimension
mismatch errors result.

### 2.3 LQR/LQE Return Type Consistency

**Source**: Issue #439, PR #477

`lqr` returned eigenvalues as 1D; `lqe` returned 2D matrix. Both should
return consistent shapes.

### 2.4 MATLAB-Validated Gramian Test Vectors

**Source**: python-control statefbk_test.py

```go
A := []float64{1, -2, 3, -4}   // 2x2 NON-SYMMETRIC (critical!)
B := []float64{5, 6, 7, 8}     // 2x2
C := []float64{4, 5, 6, 7}     // 2x2
D := []float64{13, 14, 15, 16} // 2x2

// Continuous controllability Gramian (from MATLAB):
Wc := []float64{18.5, 24.5, 24.5, 32.5}

// Continuous observability Gramian (from MATLAB):
Wo := []float64{257.5, -94.5, -94.5, 56.5}

// Also test discrete versions after c2d with dt=0.2
```

---

## 3. Lyapunov Solvers

### 3.1 Skew-Symmetric A (Eigenvalues Symmetric About Imaginary Axis)

**Source**: Issue #342

For A where eigenvalues of A and -A are close (e.g., orthogonal/skew-symmetric
matrices), the continuous Lyapunov solver (Bartels-Stewart) fails. The discrete
solver works fine for the same system.

**Test case**:
```go
// Skew-symmetric A from orthogonal group
// A and -A have shared eigenvalues
A := mat.NewDense(3, 3, []float64{
    0, -1, 0,
    1,  0, -1,
    0,  1,  0,
})
Q := eye(3)
// Lyap(A, Q) should detect this and return meaningful error
// DLyap(sqrt(0.9)*A, Q) should work fine
```

### 3.2 Workspace Reuse Pattern

Pre-allocate Lyapunov workspace for repeated solves (e.g., inside iterative
Riccati or H-infinity synthesis loops).

---

## 4. Pole Placement

### 4.1 Ackermann Return Shape

**Source**: Issue #1190, PR #1195

`place_acker` returned 1D array instead of 2D. The bug was
`K[-1, :]` (collapses dimension) vs `K[-1:, :]` (preserves 2D).

**Test case**:
```go
A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
B := mat.NewDense(2, 1, []float64{0, 1})
poles := []complex128{-1 + 1i, -1 - 1i}
K := Acker(A, B, poles)
// K should be (1, 2) not (2,)
```

### 4.2 Ackermann Must Reject MIMO

**Source**: Issue #1190, #1100

Ackermann's formula is SISO-only. Must error on MIMO input, not produce
garbage.

### 4.3 Ackermann Input Validation Bug

**Source**: Issue #1100

The python-control `acker` function validated inputs via `_ssmatrix()` but then
continued using the raw (unvalidated) `A`, `B` variables instead of the
converted `a`, `b`.

### 4.4 Wrong Poles from Old Algorithm

**Source**: Issue #117

Old Ackermann-based multi-input placement returned wrong eigenvalues.
MATLAB's `place` uses Tits-Yang robust algorithm.

**Test case** (from scipy example):
```go
A := mat.NewDense(4, 4, []float64{
    1.380, -0.2077, 6.715, -5.676,
    -0.5814, -4.290, 0, 0.6750,
    1.067, 4.273, -6.654, 5.893,
    0.0480, 4.273, 1.343, -2.104,
})
B := mat.NewDense(4, 2, []float64{
    0, 5.679,
    1.136, 1.136,
    0, 0,
    -3.146, 0,
})
P := []complex128{-0.5+1i, -0.5-1i, -5.0566, -8.6659}
K := Place(A, B, P)
// Verify: eig(A - B*K) matches P
```

### 4.5 Repeated Poles with Insufficient Rank

**Source**: python-control statefbk_test.py

```go
// 3 repeated poles but rank(B)=2 -- should fail
P := []complex128{-0.5, -0.5, -0.5, -8.6659}
// Place(A, B, P) should return error
```

### 4.6 Regression: Complex Conjugate Poles on Unstable System

**Source**: Issue #177

```go
A := mat.NewDense(2, 2, []float64{0, 1, 100, 0})  // unstable
B := mat.NewDense(2, 1, []float64{0, 1})
P := []complex128{-20+10i, -20-10i}
K := Place(A, B, P)
// Must not error with "complex pair on real eigenvalue"
```

---

## 5. Controllability & Observability

### 5.1 Input Validation for 1D Arrays

**Source**: Issue #1097, PR #1099

Passing 1D `B` (e.g., `[1, 1]` instead of column `[[1], [1]]`) produces wrong
dimensions silently.

**Test case**:
```go
A := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
B := mat.NewDense(2, 1, []float64{5, 7})
Wc := Ctrb(A, B)
// Wc should be 2x2: [B, AB] = [[5, 19], [7, 43]]
```

### 5.2 Duality Test

```go
// Ctrb(A, B) == Obsv(A', B')'
// This catches transposition bugs
```

### 5.3 Performance: Reuse Previous Power

**Source**: PR #941

Don't call `matrix_power(A, i)` each iteration. Reuse: `A^i = A * A^{i-1}`.

---

## 6. Gramians & Hankel Singular Values

### 6.1 Discrete-Time Gramians Must Use Discrete Lyapunov

**Source**: Issue #967

python-control silently used continuous Lyapunov for discrete systems.
Must use `A*X*A' - X + B*B' = 0` (discrete), not `A*X + X*A' + B*B' = 0`.

### 6.2 Cholesky-Factored Gramians: Preserve Schur Form

**Source**: Issue #1123 (open)

When computing Cholesky-factored Gramians, transposing A before passing to the
SLICOT routine destroys Schur form triangularity, forcing re-QR which
introduces epsilon errors. Use `tra='T'` flag instead of explicit transpose.

### 6.3 MATLAB-Validated HSV Test

**Source**: python-control modelsimp_test.py

```go
A := []float64{1, -2, 3, -4}   // 2x2
B := []float64{5, 7}           // 2x1
C := []float64{6, 8}           // 1x2
D := []float64{9}              // 1x1
// Hankel singular values (from MATLAB):
hsv := []float64{24.42686, 0.5731395}
```

---

## 7. Canonical Forms & Modal Decomposition

### 7.1 Repeated Eigenvalues Blow Up Eigenvector-Based Modal Form

**Source**: Issue #318, PR #495

Eigenvector matrix T becomes singular (entries ~4.5e15) for repeated
eigenvalues. **Our codebase already handles this with Schur fallback**
(commit 982472e).

### 7.2 bdschur Test Parameterization

**Source**: python-control canonical_test.py

```go
// Real distinct eigenvalues
eigvals := []complex128{-1,-2,-3,-4,-5}
// condmax=nil -> blksizes=[1,1,1,1,1]

// Force all into one block
// condmax=1.01 -> blksizes=[5]

// Repeated eigenvalues
eigvals = []complex128{-1,-1,-2,-2,-2}
// condmax=nil -> blksizes=[2,3]

// Complex pairs
eigvals = []complex128{-1+1i,-1-1i,-2+2i,-2-2i,-2}
// condmax=nil -> blksizes=[2,2,1]
```

### 7.3 Defective 2x2 Blocks

A 2x2 block `[[a, b], [-b, a]]` with near-zero `b` is defective (repeated
real eigenvalue), not a complex pair.

### 7.4 Reachable/Observable Form Validation

**Source**: python-control canonical_test.py

```go
// Start with known companion form, apply random transform, convert back
coeffs := []float64{1.0, 2.0, 3.0, 4.0, 1.0}
A_true := companion(coeffs)
T_true := randomOrthogonal(n)
A := inv(T_true) * A_true * T_true
// Canon(ss(A,B,C,D), "reachable") should recover A_true
```

Edge cases:
- Unreachable systems -> error
- Unobservable systems -> error
- MIMO systems -> error for reachable/observable forms

---

## 8. Transfer Function / State Space Conversion

### 8.1 ss2tf Produces Spurious Zeros for Ill-Conditioned Systems

**Source**: Issue #1068

```go
// DC motor with widely separated poles
// Poles at 0, -10, -990 -> coefficient range spans many orders
A := mat.NewDense(3, 3, []float64{
    0, 1, 0,
    0, -10, 1,
    0, 0, -990,
})
B := mat.NewDense(3, 1, []float64{0, 0, 1})
C := mat.NewDense(1, 3, []float64{1, 0, 0})
// ss2tf should NOT produce spurious zeros at +/-6.6e9j
```

### 8.2 MIMO tf2ss Near-Cancellation

**Source**: Issue #240

```go
// 2x2 MIMO with 1e-10 perturbation in denominator
// H11 = 0.25/(s+1), H12 = 0/(1), H21 = 0/(1), H22 = 0.25/(s+1+1e-10)
// Step response should give ~0.25 for both channels, not 0
```

### 8.3 Static Gain (Zero States)

**Source**: Issue #244, PR #129

```go
// tf2ss of static gain must produce 0-state system, not crash
tf := NewTransferFunc([]float64{23}, []float64{46})
sys := tf.StateSpace()
// sys.A should be 0x0, sys.D should be [[0.5]]
```

### 8.4 High-Order TF with Wide Coefficient Range

**Source**: Issue #935

16th-order TF with coefficients spanning 1e-31 to 1e-8 produced unstable
state-space via SLICOT (correct via scipy). Guard against high polynomial
order + wide dynamic range.

### 8.5 ss2tf with D=0 (Strictly Proper)

**Source**: Issue #156

Simple `1/(s+1)` (D=0) crashed python-control's ss2tf. Ensure strictly proper
systems are handled.

### 8.6 SIMO tf2ss Shape Mismatch

**Source**: Issue #235

Single-input, multiple-output TF conversion crashed with broadcast error in
common denominator computation.

---

## 9. System Interconnection

### 9.1 Feedback with ZPK is Numerically Dangerous

**Source**: Issue #1067, #1102

```go
// This blows up in polynomial domain (overshoot ~1e52)
sys := NewZPK([]complex128{}, []complex128{-2, -2.0/3.0}, 10.314)
cl := Feedback(sys, 1.0, -1)
// Step response should be bounded, not 1e52

// Same system via state-space works perfectly
// LESSON: Always compute feedback in state-space, not polynomial form
```

### 9.2 MIMO Scalar Multiply

**Source**: PR #1078

`2 * mimo_sys` should scale all channels, not fill all entries with 2.

### 9.3 Off-by-One in Connect

**Source**: Issue #421, PR #474

Negative feedback output indexing had boundary check bug:
`outp < 0 and -outp >= -sys.outputs` should be `-outp <= sys.outputs`.

### 9.4 MIMO Feedback Regression

**Source**: Issue #1172

MIMO feedback with numpy array controller path incorrectly converted to
TransferFunction, triggering "MIMO not implemented" error.

### 9.5 Series State Ordering vs MATLAB

**Source**: Issue #725 (closed, wontfix)

python-control's `series(sys2, sys1)` produces state vector `(x2, x1)` while
MATLAB produces `(x1, x2)`. Documented difference, not a bug.

---

## 10. Stability Margins

### 10.1 Discrete-Time Margins Need Z-Domain Treatment

**Source**: Issue #465, #523, PR #469, #566

DT systems need `z = exp(j*w*dt)` substitution, not `s = jw`. Polynomial
methods for finding |H(z)|=1 crossings are numerically fragile when
num/den coefficient magnitudes differ greatly.

**Fallback strategy** (PR #566): detect numerical inaccuracy via
`||num(z)*num(1/z)|| < 1e-3 * ||den(z)*den(1/z)||` and fall back to FRD
interpolation.

### 10.2 Gain Margin: Reciprocal Bug

**Source**: Issue #947

Gain margin was returned as magnitude (0.1) instead of margin (10 = 1/0.1).

### 10.3 Integer vs Float Input Types

**Source**: Issue #462

`margin(tf(50000, den))` (integer) gave wrong results;
`margin(tf(50000., den))` (float) was correct.

### 10.4 FRD Margin: Bad Seed for Root Finder

**Source**: Issue #61 (open)

Phase crossover frequency finder seeded with lowest frequency, converging to
wrong solution. Should seed with median frequency.

### 10.5 Multiple Crossings

**Source**: Issue #784 (open)

Only finding one gain margin when phase crosses -180 multiple times. All
crossings should be reported.

### 10.6 MATLAB-Validated Margin Test

```go
// Type 0 system with clear margins
num := []float64{1}
den := []float64{1, 3, 3, 1}  // (s+1)^3
// MATLAB: GM = 8 (18.06 dB) at wg = 1.732 rad/s
//         PM = 41.4 deg at wp = 0.767 rad/s
```

---

## 11. Frequency Response & Bode Plots

### 11.1 Phase Wrapping: Must Unwrap Like MATLAB

**Source**: Issue #467, #1179

python-control wraps phase to [-180, 180], producing discontinuous jumps.
MATLAB does continuous unwrapping.

```go
// 1/s^3 should show -270 deg (MATLAB), not +90 deg
sys := NewTransferFunc([]float64{1}, []float64{1, 0, 0, 0})
// Phase at any frequency should be -270, not +90
```

### 11.2 Phase Through Zeros: Wrong Direction

**Source**: Issue #1179 (open)

Third-order system with undamped zeros: phase jumps go in wrong direction.
MATLAB shows -90 -> +90 -> -90; python-control shows -90 -> -270 -> -450.

### 11.3 High-Order MIMO Frequency Response

**Source**: Issue #860

24x18 MIMO system: `frequency_response()` on derived TFs (H/(1+P)) returns
noisy/incorrect results due to polynomial coefficient cancellation in
high-order TF algebra. Manual complex arithmetic on state-space is correct.

**Lesson**: Evaluate frequency response in state-space domain
(`C*(jwI-A)^{-1}*B + D`), not via polynomial evaluation.

### 11.4 Phase Crossover with Poles at Origin

**Source**: Issue #1105, PR #1106

`phase_crossover_frequencies` crashes on `200/(s^3 + 21s^2 + 20s)`.

### 11.5 Bode Phase Leak Between Systems

**Source**: Issue #862

When plotting multiple systems, phase wrapping state leaked between systems,
producing different phases for identical systems.

---

## 12. Time-Domain Simulation

### 12.1 Marginally Stable Systems: Diverging Instead of Oscillating

**Source**: Issue #384

Closed-loop system with purely imaginary poles produced unstable diverging
response instead of sustained oscillation. Octave gave correct sinusoidal.

### 12.2 Integrator Step Response

**Source**: Issue #470

`(0.8s+1)/(s^2+s)` should give exponential step response starting at 0.8->1.0
but gave a ramp y=t. Pole at origin mishandled.

### 12.3 Discrete Impulse Scaling

**Source**: Issue #974

DT impulse response was scaled by extra `1/T`. Accumulator `H(z)=Tz/(z-1)`
with T=0.1: python-control gave 1.0, MATLAB/Octave gave 0.1 (correct).

### 12.4 zpk vs tf Give Different Impulse Responses

**Source**: Issue #1062

Same system defined as zpk vs tf gave unstable vs stable impulse responses.
Internal representation mismatch.

### 12.5 Delayed Dirac Input Ignored

**Source**: Issue #1152

Dirac impulse applied after index 5 completely ignored. ODE solver step size
determination skipped over the impulse.

### 12.6 MATLAB-Validated Step Response

**Source**: python-control timeresp_test.py

```go
A := []float64{1, -2, 3, -4}   // 2x2
B := []float64{5, 7}           // 2x1
C := []float64{6, 8}           // 1x2
D := []float64{9}              // 1x1
t := linspace(0, 1, 10)
// Step response (from MATLAB):
y := []float64{9., 17.6457, 24.7072, 30.4855, 35.2234,
               39.1165, 42.3227, 44.9694, 47.1599, 48.9776}
```

### 12.7 MATLAB-Validated Step Info

**Source**: python-control timeresp_test.py

```go
// G(s) = (s^2 + 5s + 5) / (s^4 + 1.65s^3 + 5s^2 + 6.5s + 2)
// From MATLAB online help:
// RiseTime: 3.8456
// SettlingTime: 27.9762
// Overshoot: 7.4915%
// SteadyStateValue: 2.5
```

### 12.8 Non-Minimum Phase Step Info

```go
// G(s) = (-s+1)/(s^2+s+1)
// Undershoot: 28%
// Overshoot: 20.84%
```

### 12.9 Overshoot: Use dcgain(), Not Final Simulated Value

**Source**: PR #555

Overshoot relative to `dcgain()`, not last simulation sample. Non-strictly
proper TFs give wrong overshoot with final value approach.

---

## 13. Discretization (c2d / d2c)

### 13.1 Matched Method DC Gain Normalization

**Source**: Issue #950, PR #951

Matched pole-zero method used polynomial leading coefficient (`sgain`) instead
of `dcgain()` for gain normalization, producing ~10x error.

**Correct approach**: Normalize so that `sys_ct(0) == sys_dt(1)`.

**Test case**:
```go
// For all discretization methods, verify:
// dcgain(c2d(sys, dt)) == dcgain(sys)
sys := NewSS(A, B, C, D)
for _, method := range []string{"zoh", "foh", "tustin", "matched"} {
    sysd := Discretize(sys, dt, method)
    assert(abs(DCGain(sysd) - DCGain(sys)) < tol)
}
```

### 13.2 Direct State-Space Discretization

**Source**: Issue #23

Never discretize by converting to TF first. Use matrix exponential
`Ad = expm(A*T)` directly.

### 13.3 ZPK Timebase Default

**Source**: PR #1064

`zpk()` defaulted timebase to `nil` (discrete) instead of 0 (continuous).
All factory functions should use consistent default.

---

## 14. Model Reduction

### 14.1 Marginally Stable Systems: Warn, Don't Error

**Source**: Issue #1166, PR #1074

Double integrator `A=[[0,1],[0,0]]` (poles at origin) was rejected as
"unstable" by balanced reduction. Should warn, not error. Eigenvalues
at exactly 0 or on imaginary axis need tolerance-aware stability check.

### 14.2 MATLAB-Validated Balanced Reduction

**Source**: python-control modelsimp_test.py

```go
// TF: num=[1,11,45,32], den=[1,15,60,200,60]
// Controllable canonical form (4 states)
// Reduced to order 2 (truncate):
Ar := []float64{-1.958, -1.194, -1.194, -0.8344}  // from MATLAB
Br := []float64{0.9057, 0.4068}
// Note: state ordering may differ -- test via similarity transform
```

### 14.3 Unstable System Warning Control

Test that `warn_unstable=false` suppresses warnings for intentionally unstable
systems.

### 14.4 Chained Transforms Corrupt Data

**Source**: Issue #1177 (open)

Combining similarity transforms + balanced reduction corrupts state data.
Test sequential operations for data integrity.

---

## 15. System Norms (H2, H-infinity)

### 15.1 MATLAB-Validated Norm Test Cases

**Source**: python-control sysnorm_test.py

```go
// 1st order stable: G = 1/(s+1)
// Hinf = 1.0, H2 = 0.707106781186547

// 1st order unstable: G = 1/(1-s)
// Hinf = 1.0, H2 = inf (with warning)

// 2nd order with imaginary poles: G = 1/(s^2+1)
// Hinf = inf (marginally stable), H2 = inf

// 3rd order MIMO (from MATLAB):
A := []float64{
    -1.017, -0.224, 0.043,
    -0.310, -0.516, -0.119,
    -1.453,  1.800, -1.492,
}  // 3x3
B := []float64{0.313, -0.165, -0.865, 0.628, -0.030, 1.093}  // 3x2
C := []float64{1.109, 0.077, -1.114, -0.864, -1.214, -0.007}  // 2x3
// Hinf = 4.276759162964244 (from MATLAB)
// H2   = 2.237461821810309 (from MATLAB)
// Also test after discretization with dt=0.1
```

### 15.2 Hinfsyn Sensitivity to Realization

**Source**: Issue #367 (open)

Same plant: MATLAB finds stabilizing H-infinity controller, python-control
says "cannot be found". Root cause: slightly different tf2ss realizations
produce numerically different state-space matrices.

**Lesson**: hinfsyn is extremely sensitive to the state-space realization.
Consider pre-balancing or using balanced realization before synthesis.

### 15.3 Hinfsyn MATLAB/Octave Reference

**Source**: python-control robust_test.py

```go
// P = ss(-1, [1,1], [[1],[1]], [[0,1],[1,0]])
// k, cl, gam, rcond = hinfsyn(P, 1, 1)
// From Octave (SB10AD):
// k.A = [[-3]], k.B = [[1]], k.C = [[-1]], k.D = [[0]]
```

---

## 16. DC Gain & Zeros

### 16.1 Poles at Origin: Per-Channel DC Gain

**Source**: Issue #128 (open)

If any eigenvalue of A is 0, `dcgain()` returns NaN for ALL outputs, even for
channels that have finite DC gain. Should compute per-channel via
`C*(s*I-A)^{-1}*B + D` evaluated at s=0 for each I/O pair.

### 16.2 Pole-Zero Cancellation at Origin

**Source**: Issue #127 (open)

Both num and den have roots at s=0: should compute the limit (ratio of lowest
non-zero coefficients), not return NaN.

### 16.3 Consistent Evaluation at s=0

**Source**: Issue #532

`tf([1],[1,0])(0)` returns `inf+0j`; SS version raises `LinAlgError`.
`dcgain()` returns inf vs nan depending on representation. Must be consistent.

### 16.4 Non-Square MIMO Zeros

**Source**: Issue #859

Transmission zeros for non-square MIMO (e.g., 3-in 6-out) require generalized
eigenvalue approach that handles rectangular matrices.

### 16.5 Small Real Parts from Roundoff

**Source**: Issue #768

Zeros reported as `-4.72e-16 +/- 1j` instead of clean `+/- 1j`. Inherent
floating-point limitation -- consider optional cleanup threshold.

---

## 17. Nyquist Plots

### 17.1 Indentation Around Imaginary-Axis Poles

**Source**: PR #722

Default indentation radius (0.1) around poles near imaginary axis could miss
right-half-plane closed-loop poles. Reduced to 1e-4.

### 17.2 DT Poles at z=0 and z=1

**Source**: PR #885

Nyquist plot failed for DT TFs with poles at z=0 and z=1. Need special
handling for these discrete-time singularities.

### 17.3 Indentation Direction

**Source**: Issue #1191 (open)

Indentation curves around imaginary-axis poles go counter-clockwise instead of
standard clockwise.

---

## 18. Numerical Primitives & General Patterns

### 18.1 Use solve(), Never inv()

**Source**: PR #91, #101, #200

Replace ALL `inv(M) * X` with `solve(M, X)`. This applies to:
- Riccati gain computation
- Canonical form transforms
- Feedback computation
- DC gain (C * inv(-A) * B)

### 18.2 Use eigvals(), Never roots(poly(A))

**Source**: PR #91, Issue #84

Computing poles via `det(sI-A)` -> roots is O(n!) worse numerically than
direct eigenvalue decomposition.

### 18.3 Separate Real/Imaginary Tolerances for Pole Matching

**Source**: PR #345

```go
realTol := math.Sqrt(eps * float64(nInputs*nOutputs))
imagTol := 2 * realTol
// Poles with |imag| < imagTol are forced to real
```

### 18.4 Complex-to-Real Casting After Polynomial Reconstruction

**Source**: PR #1086

After reconstructing polynomials from complex poles/zeros, cast coefficients
to real to avoid spurious complex warnings.

### 18.5 Non-Symmetric A Matrices Are Critical for Testing

**Source**: CLAUDE.md, python-control test suite

Nearly every test suite uses `A = [[1, -2], [3, -4]]` or similar
non-symmetric matrices. Diagonal or symmetric A hides transposition bugs.

### 18.6 Replace det() Singularity Check with rank()

**Source**: PR #91, #101

`abs(det(F)) < threshold` is unreliable. Use `matrix_rank()` (SVD-based with
proper numerical threshold) instead.

### 18.7 damp() for DT: Cast to Complex Before Log

**Source**: Issue #646

Discrete-time systems with negative poles: `log(poles)` fails for real
negative values. Must cast to complex first: `log(complex(pole))`.

### 18.8 StateSpace Constructor: Never Auto-Reduce

**Source**: Issue #244

Don't call `_remove_useless_states()` in the constructor. `ss(0,0,1,0)` must
preserve the single state, not collapse to empty.

### 18.9 Overshoot Relative to dcgain(), Not Simulation End

**Source**: PR #555

For non-strictly proper systems, the final simulated value != DC gain.
Use `dcgain()` as the reference for percent overshoot calculation.

---

## Appendix A: Test System Zoo

Standard test systems used across python-control (with MATLAB reference values):

| Name | A | B | C | D | Notes |
|------|---|---|---|---|-------|
| **Basic 2x2** | `[[1,-2],[3,-4]]` | `[[5],[7]]` | `[[6,8]]` | `[[9]]` | Non-symmetric A, primary test system |
| **Basic 2x2 MIMO** | `[[1,-2],[3,-4]]` | `[[5,6],[7,8]]` | `[[4,5],[6,7]]` | `[[13,14],[15,16]]` | MIMO variant |
| **Double integrator** | `[[0,1],[0,0]]` | `[[0],[1]]` | `[[1,0]]` | `[[0]]` | Marginally stable, poles at 0 |
| **Unstable** | `[[0,1],[100,0]]` | `[[0],[1]]` | `[[1,0]]` | `[[0]]` | Unstable open-loop |
| **1/(s+1)** | `[[-1]]` | `[[1]]` | `[[1]]` | `[[0]]` | Hinf=1, H2=0.7071 |
| **1/(s^2+1)** | `[[0,1],[-1,0]]` | `[[0],[1]]` | `[[1,0]]` | `[[0]]` | Marginally stable, imaginary poles |
| **DC motor** | 3x3, poles at 0,-10,-990 | | | | Wide pole spread |
| **4-state (scipy)** | 4x4 dense | 4x2 | | | Pole placement reference |

## Appendix B: Priority Matrix

| Priority | Area | Issue | Our Status |
|----------|------|-------|------------|
| **DONE** | Modal form | Schur fallback for repeated eigenvalues | Commit 982472e |
| **DONE** | Input validation | Improper TF, descriptor Riccati, Q PSD | Commit 44d7971 |
| **DONE** | Numerical hardening | Python-control PR audit fixes | Commit 2392746 |
| HIGH | Riccati | Scale-aware symmetry check | Verify current impl |
| HIGH | DARE | Ill-conditioned A robustness | Test coverage |
| HIGH | Margins | DT margin computation | Test coverage |
| HIGH | Feedback | ZPK/polynomial blow-up | Test with pathological gains |
| HIGH | ss2tf | Spurious zeros for ill-conditioned systems | Test coverage |
| MED | dlqe | Filter vs predictor gain convention | Document + test |
| MED | Gramians | DT Gramian correctness | Test coverage |
| MED | Model reduction | Marginally stable handling | Test coverage |
| MED | DC gain | Poles at origin, per-channel | Test + fix |
| MED | Phase | Unwrapping like MATLAB | Test coverage |
| LOW | Acker | Return shape, MIMO rejection | Verify |
| LOW | c2d matched | DC gain normalization | Test coverage |
| LOW | Norms | Discrete-time H2/Hinf | MATLAB-validated tests |

## Appendix C: Key python-control PRs for Reference

| PR | Title | Key Pattern |
|----|-------|-------------|
| #91 | eigvals for poles, solve for feedback | Never use roots(poly(A)) |
| #101 | Replace inv() with solve() | Numerical stability |
| #200 | Fix observable canonical form | solve() + rank check |
| #206 | _common_den rewrite | Pole matching algorithm |
| #345 | Root precision tolerance | Separate real/imag tolerances |
| #348 | Machine-epsilon symmetry check | Scale-aware tolerance |
| #469 | DT stability margins | z-domain treatment |
| #495 | bdschur for modal form | Condition number control |
| #566 | Margin fallback to FRD | Auto-detect numerical issues |
| #683 | LQR using scipy | Dual-method approach |
| #951 | Matched c2d DC gain | Use dcgain() not leading coeff |
| #1078 | MIMO scalar multiply | Element-wise, not filled matrix |
| #1195 | place_acker return shape | Preserve 2D with [-1:, :] |

## Appendix D: Real-World Systems from Jupyter Notebooks

Extracted from 14 Caltech CDS 110/112 course notebooks. These are physically
meaningful systems that stress the library differently than toy 2x2 examples.

### D.1 PVTOL Aircraft (6-state, 2-input MIMO)

The most exercised system across notebooks. Parameters: m=4, J=0.0475,
r=0.25, g=9.8, c=0.05.

Linearized at hover (xe=0, ue=[0, m*g]):

```go
A := []float64{
    0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 1,
    0, 0, -9.8, -0.0125, 0, 0,
    0, 0, 0, 0, -0.0125, 0,
    0, 0, 0, 0, 0, 0,
}  // 6x6
B := []float64{
    0, 0,
    0, 0,
    0, 0,
    0.25, 0,
    0, 0.25,
    5.2632, 0,
}  // 6x2, r/J=5.2632
C_pos := eye(3, 6)  // measures x, y, theta only
```

**LQR weights tested** (3 cases):
- Identity: Qx=I(6), Qu=I(2)
- Physical: Qx=diag(100, 10, 180/pi/5, 0, 0, 0), Qu=diag(10, 1)
- High-drag (c=20): Qx=diag(10, 100, 180/pi/5, 0, 0, 0), Qu=diag(10, 1)

**Kalman with non-diagonal Qw**:
```go
Qv := []float64{1e-2, 0, 0, 1e-2}  // 2x2 disturbance
Qw := []float64{
    2e-4, 0,    1e-5,
    0,    2e-4, 1e-5,
    1e-5, 1e-5, 1e-4,
}  // 3x3 measurement noise, cross-terms!
```

**Why valuable**: Open-loop unstable (double integrator in theta), MIMO,
real aircraft parameters, exercises LQR + Kalman + margins simultaneously.

### D.2 Coupled Spring-Mass (4-state, 2-output)

m=1, c=0.1, k=2.

```go
A := []float64{
    0, 0, 1, 0,
    0, 0, 0, 1,
    -4, 2, -0.1, 0,
    2, -4, 0, -0.1,
}  // 4x4
B := []float64{0, 0, 0, 2}  // 4x1
C := []float64{1, 0, 0, 0, 0, 1, 0, 0}  // 2x4
```

**Why valuable**: Has transmission zero (dip in frequency response of q2
at ~2 rad/s). Lightly damped. Good for SIMO transfer function and
frequency response testing.

### D.3 Inverted Pendulum (unstable SISO)

```go
// P(s) = 1/(s^2 + 0.1s - 1), one RHP pole
A := []float64{0, 1, 1, -0.1}  // 2x2
B := []float64{0, 1}
C := []float64{1, 0}
D := []float64{0}
```

**Why valuable**: Open-loop unstable. Requires Nyquist encirclement
N=-1 for closed-loop stability. With PD controller kp=10, kd=2:
`L(s) = (2s+10)/(s^2+0.1s-1)`.

### D.4 Maglev System (unstable, real hardware)

Caltech maglev experiment. Parameters: m=0.2, g=9.81, km=3.13e-4.

2-state, 1-input, 1-output. Open-loop unstable (one RHP pole).

Analog controller (RC circuit):
```go
// C(s) = -0.5 * (0.024s + 1)/(0.002s + 1)
// Lead compensator from actual hardware
```

**Why valuable**: Real hardware parameters. Tests Bode sensitivity
integral: `integral(ln|S(jw)|, 0, inf) = pi * sum(Re(unstable_poles))`.
Demonstrates fundamental performance limits from RHP poles.

### D.5 Vehicle Steering (double integrator + observer)

Bicycle model at v0=15 m/s, normalized to double integrator:

```go
A := []float64{0, 1, 0, 0}  // 2x2
B := []float64{0, 1}
C := []float64{1, 0}
```

**Placement tested**:
- Feedback: wc=0.7, zeta=0.707 (standard) and wc=10 (high bandwidth)
- Observer: wo=1, zeta=0.7
- Observer gain via duality: `L = Place(A', C', obs_poles)'`

**Why valuable**: Tests observer+controller design, duality for observer
gain. Also tested in reverse (v=-2 m/s) which creates non-minimum phase
dynamics.

### D.6 Predator-Prey (unstable complex poles)

Lotka-Volterra: r=1.6, d=0.56, b=0.6, k=125, a=3.2, c=50.
2-state, 1-input. Linearized at interior equilibrium (~[21.05, 24.61]).

**Why valuable**: Unstable complex eigenvalues from biological system.
Demonstrates sensitivity: changing r from 1.6 to 1.65 (<4%) destabilizes
proportional controller, requiring integral action (Ki=0.0001).

### D.7 Cruise Control (nonlinear 1-state)

m=1600 kg, Tm=190 N-m. Single velocity state.
Linearized: first-order P(s)=b/(s+a) at v=20 m/s.

```go
// State feedback K=0.5, feedforward kf = -1/(C*inv(A-BK)*B)
// Integral gain ki=0.1
// Robustness test: m varies from 1200 to 2000 kg
```

**Why valuable**: Simplest nonlinear system. Tests robustness to
parameter uncertainty (25% mass variation).

### D.8 Servomechanism (margins edge case)

J=100, b=10, k=1. Third-order with actuator.

```go
// PI that creates UNSTABLE closed loop (Nyquist encirclement):
// C(s) = (s+1)/s, with plant P(s) = 0.02/(s^2 + 0.1s + 0.00966)

// PI with reduced integral gain (stable):
// C(s) = (s+0.005)/s

// PID (loop-shaped):
// kp=150, ki=30, kd=150
```

**Why valuable**: Same plant produces unstable CL with kp=ki=1 but stable
CL with ki=0.005. Tests that margin computation correctly identifies
instability. Ziegler-Nichols tuning also demonstrated.

## Appendix E: Example Scripts — Textbook Reference Values

Extracted from Python scripts in `examples/` folder. These provide
published/textbook numerical values for verification.

### E.1 Scherer et al. H-infinity / H2 Synthesis (Gold Standard)

**Reference**: Scherer, Gahinet & Chilali, "Multiobjective Output-Feedback
Control via LMI Optimization", IEEE TAC, Vol. 42, No. 7, July 1997, Example 7.

```go
A := []float64{
    0, 10, 2,
    -1, 1, 0,
    0, 2, -5,
}  // 3x3, open-loop UNSTABLE (eigs ≈ -5.37, 1.69±2.50j)

B1 := []float64{1, 0, 1}     // 3x1 disturbance
B2 := []float64{0, 1, 0}     // 3x1 control

// H2 performance output
C1_h2 := []float64{0, 1, 0, 0, 0, 1, 0, 0, 0}  // 3x3
D12_h2 := []float64{0, 0, 1}                      // 3x1

// Measurement
C2 := []float64{0, 1, 0}     // 1x3
D21 := []float64{2}          // 1x1
```

**H2 synthesis**: `h2syn(P, nmeas=1, ncon=1)`
**Published H2 norm = 7.748350599360575**

**H-infinity synthesis**: Same plant, different C1/D12:
```go
C1_hinf := []float64{1, 0, 0, 0, 0, 0}  // 2x3
D12_hinf := []float64{0, 1}              // 2x1
```

### E.2 Skogestad Mixed-Sensitivity MIMO (Textbook Gammas)

**Reference**: Skogestad & Postlethwaite, Example 3.8, 1st Ed.

```go
// 2x2 MIMO plant with RHP zero at s=0.5
// den = 0.2s^2 + 1.2s + 1
// G = [[1/den, 1/den], [(2s+1)/den, 2/den]]

// Sensitivity weight: W(s) = (s/M + wb) / (s + wb*A)
// M=1.5, A=1e-4
```

**Published gammas from textbook**:
- wb1=0.25, wb2=0.25 → **gamma ≈ 2.80**
- wb1=0.25, wb2=25   → **gamma ≈ 2.92**
- wb1=25,   wb2=0.25 → **gamma ≈ 6.73**

### E.3 Disk Margins Test Systems

**SISO system 1**:
```go
// L(s) = 25 / (s^3 + 10s^2 + 10s + 10)
A := []float64{0, 1, 0, 0, 0, 1, -10, -10, -10}  // 3x3 companion
B := []float64{0, 0, 1}
C := []float64{25, 0, 0}
```
Test with skew = {-1, 0, 1} (T-based, balanced, S-based).

**SISO system 2** (5th order, more complex):
```go
// L(s) = 6.25(s+3)(s+5) / (s(s+1)^2(s^2+0.18s+100))
// num = [6.25, 50, 93.75]
// den = [1, 2.18, 100.36, 200.18, 100, 0]
```

**MIMO disk margins**:
```go
// Plant: purely oscillatory
A := []float64{0, 10, -10, 0}  // 2x2, eigs at ±10j
B := eye(2)
C := []float64{1, 10, -10, 1}  // 2x2
D := zeros(2, 2)
// Static controller K = [[1,-2],[0,1]]
```

### E.4 ERA System Identification (2-DOF MSD)

**Reference**: Mechanical Vibrations textbook, Ex 6.7.

```go
A := []float64{
    0,  0,  1,    0,
    0,  0,  0,    1,
    -6, 2,  -2,   1,
    1,  -4, 0.5,  -1.5,
}  // 4x4
B := []float64{0, 0, 0, 0, 1, 0, 0, 0.5}  // 4x2
C := []float64{1, 0, 0, 0, 0, 1, 0, 0}    // 2x4
D := zeros(2, 2)
dt := 0.1
```

Recover 4-state system from impulse response via `ERA(markov, r=4, dt=0.1)`.
Verify: recovered poles match original, frequency response matches.

### E.5 Balanced Reduction (Deterministic, MATLAB-Origin)

```go
A := []float64{
    -15, -7.5, -6.25, -1.875,
    8,    0,    0,     0,
    0,    4,    0,     0,
    0,    0,    1,     0,
}  // 4x4 controllable canonical form
B := []float64{2, 0, 0, 0}              // 4x1
C := []float64{0.5, 0.6875, 0.7031, 0.5}  // 1x4
D := []float64{0}
// TF: (s^3+11s^2+45s+32) / (s^4+15s^3+60s^2+200s+60)
// DC gain = 32/60 = 0.5333
```

Reduce order 4 → 3 (truncate). Verify DC gain preserved.
Reduce order 4 → 2 (truncate). From MATLAB:
```go
Ar := []float64{-1.958, -1.194, -1.194, -0.8344}  // 2x2
Br := []float64{0.9057, 0.4068}                    // 2x1
```

### E.6 Type 2/3 Systems (Margin Edge Cases)

Multiple integrators in the loop — stress test for margin computation.

```go
// Plant: pure integrator P(s) = 1/s with friction b=10
// Inner plant: Peff(s) = (1/s) / (1 + 10/s) = 1/(s+10)

// Type 2 controller: C(s) = 165*(s+55)/s
// Loop has 1 integrator from plant → type 1; +1 from controller → type 2

// Type 3 controller: C(s) = 110*((s+55)/s)^2
// Loop has 1 integrator from plant → type 1; +2 from controller → type 3
// Phase at DC = -270° (3 integrators), margin computation must handle this
```

### E.7 Block Diagram Algebra (SS↔TF Roundtrip)

```go
// System 1 (state-space):
A1 := []float64{0, 1, -4, -1}  // underdamped, wn=2, zeta=0.25
B1 := []float64{0, 1}
C1 := []float64{1, 0}
// TF: G1(s) = 1/(s^2+s+4)

// System 2 (transfer function):
// G2(s) = (s+0.5)/(s+5)

// Verify: Series, Parallel, Feedback produce correct poles
```

### E.8 Second-Order Reference (wn, zeta verification)

```go
// Mass-spring-damper: m=250, k=40, b=60
A := []float64{0, 1, -0.16, -0.24}  // 2x2
B := []float64{0, 0.004}
C := []float64{1, 0}
// wn = sqrt(0.16) = 0.4 rad/s
// zeta = 0.24/(2*0.4) = 0.3
// Verify Damp() returns these values
```

### E.9 Controllability/Observability (3-State with Physical Params)

```go
A := []float64{1, -1, 1, 1, -0.16, -0.24, 1, 1, 1}  // 3x3
B := []float64{0, 0.004, 1}                            // 3x1
C := []float64{1, 0, 1}                                // 1x3
// Ctrb = [B, AB, A^2*B]  → 3x3, check rank
// Obsv = [C; CA; CA^2]   → 3x3, check rank
```
