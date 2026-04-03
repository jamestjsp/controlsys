package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/mat"
)

type TimeResponse struct {
	T          []float64
	Y          *mat.Dense
	OutputName []string
}

type DampInfo struct {
	Pole complex128
	Wn   float64
	Zeta float64
	Tau  float64
}

func autoTimeParams(sys *System) (dt, tFinal float64, err error) {
	n, _, _ := sys.Dims()
	if n == 0 {
		if sys.IsDiscrete() {
			return sys.Dt, 100 * sys.Dt, nil
		}
		return 0.01, 1.0, nil
	}

	poles, err := sys.Poles()
	if err != nil {
		return 0, 0, err
	}

	var wns []float64
	for _, p := range poles {
		var wn float64
		if sys.IsContinuous() {
			wn = cmplx.Abs(p)
		} else {
			lp := cmplx.Log(p)
			wn = cmplx.Abs(lp) / sys.Dt
		}
		if wn > 1e-10 {
			wns = append(wns, wn)
		}
	}

	if len(wns) == 0 {
		if sys.IsDiscrete() {
			return sys.Dt, 100 * sys.Dt, nil
		}
		return 0.01, 1.0, nil
	}

	minWn, maxWn := wns[0], wns[0]
	for _, w := range wns[1:] {
		if w < minWn {
			minWn = w
		}
		if w > maxWn {
			maxWn = w
		}
	}

	tFinal = 7.0 / minWn
	if tFinal > 1e4 {
		tFinal = 1e4
	}

	if sys.IsDiscrete() {
		dt = sys.Dt
	} else {
		dt = 1.0 / (20 * maxWn)
		if maxDt := tFinal / 100; dt > maxDt {
			dt = maxDt
		}
	}

	return dt, tFinal, nil
}

func prepareDiscrete(sys *System, tFinal, dt float64) (dsys *System, actualDt float64, steps int, wasContinuous bool, err error) {
	if sys.IsDiscrete() {
		actualDt = sys.Dt
		if tFinal <= 0 {
			_, tFinal, err = autoTimeParams(sys)
			if err != nil {
				return nil, 0, 0, false, err
			}
		}
		steps = int(tFinal/actualDt) + 1
		return sys, actualDt, steps, false, nil
	}

	wasContinuous = true
	if tFinal <= 0 || dt <= 0 {
		autoDt, autoTf, err2 := autoTimeParams(sys)
		if err2 != nil {
			return nil, 0, 0, true, err2
		}
		if tFinal <= 0 {
			tFinal = autoTf
		}
		if dt <= 0 {
			dt = autoDt
		}
	}

	dsys, err = sys.DiscretizeZOH(dt)
	if err != nil {
		return nil, 0, 0, true, fmt.Errorf("auto-discretize: %w", err)
	}
	actualDt = dt
	steps = int(tFinal/dt) + 1
	return dsys, actualDt, steps, true, nil
}

func makeTimeVector(steps int, dt float64) []float64 {
	t := make([]float64, steps)
	for k := range t {
		t[k] = float64(k) * dt
	}
	return t
}

func (sys *System) DCGain() (*mat.Dense, error) {
	n, m, p := sys.Dims()

	if n == 0 {
		return denseCopy(sys.D), nil
	}

	if sys.IsContinuous() {
		var X mat.Dense
		err := X.Solve(sys.A, sys.B)
		if err != nil {
			return nil, fmt.Errorf("DCGain: A singular, DC gain undefined: %w", ErrSingularTransform)
		}
		gain := mat.NewDense(p, m, nil)
		gain.Mul(sys.C, &X)
		gain.Scale(-1, gain)
		if sys.D != nil {
			gain.Add(gain, sys.D)
		}
		return gain, nil
	}

	ImA := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			v := -sys.A.At(i, j)
			if i == j {
				v += 1
			}
			ImA.Set(i, j, v)
		}
	}

	var X mat.Dense
	err := X.Solve(ImA, sys.B)
	if err != nil {
		return nil, fmt.Errorf("DCGain: (I-A) singular, DC gain undefined: %w", ErrSingularTransform)
	}
	gain := mat.NewDense(p, m, nil)
	gain.Mul(sys.C, &X)
	if sys.D != nil {
		gain.Add(gain, sys.D)
	}
	return gain, nil
}

func Damp(sys *System) ([]DampInfo, error) {
	poles, err := sys.Poles()
	if err != nil {
		return nil, err
	}
	if len(poles) == 0 {
		return nil, nil
	}

	result := make([]DampInfo, len(poles))
	for i, p := range poles {
		var wn, zeta float64

		if sys.IsDiscrete() {
			sc := cmplx.Log(p) / complex(sys.Dt, 0)
			wn = cmplx.Abs(sc)
			if wn > 0 {
				zeta = -real(sc) / wn
			}
		} else {
			wn = cmplx.Abs(p)
			if wn > 0 {
				zeta = -real(p) / wn
			}
		}

		tau := math.Inf(1)
		if sigma := zeta * wn; sigma > 0 {
			tau = 1.0 / sigma
		}

		result[i] = DampInfo{Pole: p, Wn: wn, Zeta: zeta, Tau: tau}
	}
	return result, nil
}

func Step(sys *System, tFinal float64) (*TimeResponse, error) {
	dsys, dt, steps, _, err := prepareDiscrete(sys, tFinal, 0)
	if err != nil {
		return nil, err
	}

	_, m, p := dsys.Dims()
	rows := p * m
	Y := mat.NewDense(rows, steps, nil)

	for j := 0; j < m; j++ {
		u := mat.NewDense(m, steps, nil)
		for k := 0; k < steps; k++ {
			u.Set(j, k, 1)
		}
		resp, err := dsys.Simulate(u, nil, nil)
		if err != nil {
			return nil, fmt.Errorf("Step: input %d: %w", j, err)
		}
		if resp.Y != nil {
			for i := 0; i < p; i++ {
				for k := 0; k < steps; k++ {
					Y.Set(j*p+i, k, resp.Y.At(i, k))
				}
			}
		}
	}

	return &TimeResponse{T: makeTimeVector(steps, dt), Y: Y, OutputName: copyStringSlice(sys.OutputName)}, nil
}

func Impulse(sys *System, tFinal float64) (*TimeResponse, error) {
	dsys, dt, steps, wasContinuous, err := prepareDiscrete(sys, tFinal, 0)
	if err != nil {
		return nil, err
	}

	_, m, p := dsys.Dims()
	rows := p * m
	Y := mat.NewDense(rows, steps, nil)

	amp := 1.0
	if wasContinuous {
		amp = 1.0 / dt
	}

	for j := 0; j < m; j++ {
		u := mat.NewDense(m, steps, nil)
		u.Set(j, 0, amp)
		resp, err := dsys.Simulate(u, nil, nil)
		if err != nil {
			return nil, fmt.Errorf("Impulse: input %d: %w", j, err)
		}
		if resp.Y != nil {
			for i := 0; i < p; i++ {
				for k := 0; k < steps; k++ {
					Y.Set(j*p+i, k, resp.Y.At(i, k))
				}
			}
		}
	}

	return &TimeResponse{T: makeTimeVector(steps, dt), Y: Y, OutputName: copyStringSlice(sys.OutputName)}, nil
}

func Initial(sys *System, x0 *mat.VecDense, tFinal float64) (*TimeResponse, error) {
	if x0 == nil {
		return nil, fmt.Errorf("Initial: x0 must not be nil: %w", ErrDimensionMismatch)
	}

	dsys, dt, steps, _, err := prepareDiscrete(sys, tFinal, 0)
	if err != nil {
		return nil, err
	}

	_, m, _ := dsys.Dims()
	u := mat.NewDense(m, steps, nil)
	resp, err := dsys.Simulate(u, x0, nil)
	if err != nil {
		return nil, fmt.Errorf("Initial: %w", err)
	}

	return &TimeResponse{T: makeTimeVector(steps, dt), Y: resp.Y, OutputName: copyStringSlice(sys.OutputName)}, nil
}

func Lsim(sys *System, u *mat.Dense, t []float64, x0 *mat.VecDense) (*TimeResponse, error) {
	if len(t) < 2 {
		return nil, fmt.Errorf("Lsim: need at least 2 time points: %w", ErrDimensionMismatch)
	}

	_, m, _ := sys.Dims()
	ur, uc := u.Dims()
	if ur != len(t) || uc != m {
		return nil, fmt.Errorf("Lsim: u must be %d×%d, got %d×%d: %w", len(t), m, ur, uc, ErrDimensionMismatch)
	}

	steps := len(t)
	dt := t[1] - t[0]
	if dt <= 0 {
		return nil, fmt.Errorf("Lsim: time step must be positive, got %g: %w", dt, ErrDimensionMismatch)
	}
	for k := 2; k < steps; k++ {
		dk := t[k] - t[k-1]
		if math.Abs(dk-dt)/dt > 1e-6 {
			return nil, fmt.Errorf("Lsim: non-uniform time grid at index %d (dt=%g, expected %g); uniform grid required: %w", k, dk, dt, ErrDimensionMismatch)
		}
	}

	var dsys *System
	var err error
	if sys.IsContinuous() {
		dsys, err = sys.DiscretizeZOH(dt)
		if err != nil {
			return nil, fmt.Errorf("Lsim: %w", err)
		}
	} else {
		dsys = sys
	}

	// Transpose u from len(t)×m to m×len(t) for Simulate
	uSim := mat.NewDense(m, steps, nil)
	uRaw := u.RawMatrix()
	uSimRaw := uSim.RawMatrix()
	for i := 0; i < steps; i++ {
		for j := 0; j < m; j++ {
			uSimRaw.Data[j*uSimRaw.Stride+i] = uRaw.Data[i*uRaw.Stride+j]
		}
	}

	resp, err := dsys.Simulate(uSim, x0, nil)
	if err != nil {
		return nil, fmt.Errorf("Lsim: %w", err)
	}

	return &TimeResponse{T: t, Y: resp.Y, OutputName: copyStringSlice(sys.OutputName)}, nil
}
