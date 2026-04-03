package controlsys

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type PIDForm int

const (
	PIDParallel PIDForm = iota
	PIDStandard
)

type PID struct {
	Kp   float64
	Ki   float64
	Kd   float64
	Tf   float64
	Dt   float64
	Form PIDForm
}

type PIDOption func(*PID)

func WithFilter(Tf float64) PIDOption {
	return func(p *PID) { p.Tf = Tf }
}

func WithTs(dt float64) PIDOption {
	return func(p *PID) { p.Dt = dt }
}

func NewPID(Kp, Ki, Kd float64, opts ...PIDOption) *PID {
	p := &PID{Kp: Kp, Ki: Ki, Kd: Kd, Form: PIDParallel}
	for _, o := range opts {
		o(p)
	}
	return p
}

// NewPIDStd creates a PID in standard (ISA) form:
//
//	C(s) = Kp * (1 + 1/(Ti*s) + Td*s/(Tf*s + 1))
//
// Relation to parallel: Ki = Kp/Ti, Kd = Kp*Td.
func NewPIDStd(Kp, Ti, Td float64, opts ...PIDOption) (*PID, error) {
	if Ti == 0 && Kp != 0 {
		return nil, fmt.Errorf("controlsys: Ti must be nonzero in standard form")
	}
	var Ki, Kd float64
	if Ti != 0 {
		Ki = Kp / Ti
	}
	Kd = Kp * Td
	p := &PID{Kp: Kp, Ki: Ki, Kd: Kd, Form: PIDStandard}
	for _, o := range opts {
		o(p)
	}
	return p, nil
}

// Parallel returns a copy in parallel form (Kp, Ki, Kd).
func (p *PID) Parallel() *PID {
	return &PID{Kp: p.Kp, Ki: p.Ki, Kd: p.Kd, Tf: p.Tf, Dt: p.Dt, Form: PIDParallel}
}

// Standard returns a copy in standard form (Kp, Ti, Td).
// Ti = Kp/Ki, Td = Kd/Kp. Requires Kp != 0 and Ki != 0.
func (p *PID) Standard() (*PID, error) {
	if p.Kp == 0 {
		return nil, fmt.Errorf("controlsys: Kp must be nonzero for standard form")
	}
	if p.Ki == 0 {
		return nil, fmt.Errorf("controlsys: Ki must be nonzero for standard form (Ti would be Inf)")
	}
	return &PID{Kp: p.Kp, Ki: p.Ki, Kd: p.Kd, Tf: p.Tf, Dt: p.Dt, Form: PIDStandard}, nil
}

// Ti returns the integral time constant (standard form). Returns 0 if Ki=0.
func (p *PID) Ti() float64 {
	if p.Ki == 0 {
		return 0
	}
	return p.Kp / p.Ki
}

// Td returns the derivative time constant (standard form). Returns 0 if Kp=0.
func (p *PID) Td() float64 {
	if p.Kp == 0 {
		return 0
	}
	return p.Kd / p.Kp
}

// PID2 represents a 2-DOF PID controller.
//
//	u = Kp*(b*r - y) + Ki/s*(r - y) + Kd*s/(Tf*s+1)*(c*r - y)
//
// The System() method produces a 2-input (r, y) to 1-output (u) system.
type PID2 struct {
	Kp float64
	Ki float64
	Kd float64
	Tf float64
	B  float64 // setpoint weight on proportional
	C  float64 // setpoint weight on derivative
	Dt float64
}

func NewPID2(Kp, Ki, Kd, Tf, b, c float64, opts ...PIDOption) *PID2 {
	p2 := &PID2{Kp: Kp, Ki: Ki, Kd: Kd, Tf: Tf, B: b, C: c}
	tmp := &PID{Dt: p2.Dt}
	for _, o := range opts {
		o(tmp)
	}
	p2.Dt = tmp.Dt
	return p2
}

// System converts the 2-DOF PID to a 2-input (r,y) 1-output (u) state-space.
func (p *PID2) System() (*System, error) {
	if p.Kd != 0 && p.Tf == 0 {
		return nil, fmt.Errorf("controlsys: 2-DOF PD without filter (Tf=0) is improper; set Tf > 0")
	}

	hasI := p.Ki != 0
	hasD := p.Kd != 0 && p.Tf != 0

	n := 0
	if hasI {
		n++
	}
	if hasD {
		n++
	}

	if n == 0 {
		D := mat.NewDense(1, 2, []float64{p.Kp * p.B, -p.Kp})
		return NewGain(D, p.Dt)
	}

	A := mat.NewDense(n, n, nil)
	Bmat := mat.NewDense(n, 2, nil)
	Cmat := mat.NewDense(1, n, nil)
	Dmat := mat.NewDense(1, 2, nil)

	idx := 0
	dFeedR := p.Kp * p.B
	dFeedY := -p.Kp

	if hasI {
		A.Set(idx, idx, 0)
		Bmat.Set(idx, 0, 1)
		Bmat.Set(idx, 1, -1)
		Cmat.Set(0, idx, p.Ki)
		idx++
	}

	if hasD {
		invTf := 1.0 / p.Tf
		A.Set(idx, idx, -invTf)
		Bmat.Set(idx, 0, p.C*invTf)
		Bmat.Set(idx, 1, -invTf)
		Cmat.Set(0, idx, -p.Kd*invTf)
		dFeedR += p.Kd * invTf * p.C
		dFeedY += -p.Kd * invTf
	}

	Dmat.Set(0, 0, dFeedR)
	Dmat.Set(0, 1, dFeedY)

	if p.Dt > 0 {
		return p.discrete2DOF(n, hasI, hasD)
	}

	return New(A, Bmat, Cmat, Dmat, 0)
}

func (p *PID2) discrete2DOF(n int, hasI, hasD bool) (*System, error) {
	dt := p.Dt
	A := mat.NewDense(n, n, nil)
	Bmat := mat.NewDense(n, 2, nil)
	Cmat := mat.NewDense(1, n, nil)
	Dmat := mat.NewDense(1, 2, nil)

	idx := 0
	dFeedR := p.Kp * p.B
	dFeedY := -p.Kp

	if hasI {
		A.Set(idx, idx, 1)
		Bmat.Set(idx, 0, dt)
		Bmat.Set(idx, 1, -dt)
		Cmat.Set(0, idx, p.Ki)
		idx++
	}

	if hasD {
		alpha := 1.0 - dt/p.Tf
		invTf := 1.0 / p.Tf
		A.Set(idx, idx, alpha)
		Bmat.Set(idx, 0, dt*invTf*p.C)
		Bmat.Set(idx, 1, -dt*invTf)
		Cmat.Set(0, idx, -p.Kd*invTf)
		dFeedR += p.Kd * invTf * p.C
		dFeedY += -p.Kd * invTf
	}

	Dmat.Set(0, 0, dFeedR)
	Dmat.Set(0, 1, dFeedY)

	return New(A, Bmat, Cmat, Dmat, dt)
}

func (p *PID) System() (*System, error) {
	if p.Kd != 0 && p.Tf == 0 && p.Ki == 0 {
		return nil, fmt.Errorf("controlsys: PD without filter (Tf=0) is improper; set Tf > 0")
	}

	if p.Dt > 0 {
		return p.discreteSystem()
	}
	return p.continuousSystem()
}

func (p *PID) continuousSystem() (*System, error) {
	hasI := p.Ki != 0
	hasD := p.Kd != 0 && p.Tf != 0

	switch {
	case !hasI && !hasD:
		return NewGain(mat.NewDense(1, 1, []float64{p.Kp}), 0)

	case hasI && !hasD:
		return New(
			mat.NewDense(1, 1, []float64{0}),
			mat.NewDense(1, 1, []float64{1}),
			mat.NewDense(1, 1, []float64{p.Ki}),
			mat.NewDense(1, 1, []float64{p.Kp}),
			0,
		)

	case !hasI && hasD:
		invTf := 1.0 / p.Tf
		return New(
			mat.NewDense(1, 1, []float64{-invTf}),
			mat.NewDense(1, 1, []float64{invTf}),
			mat.NewDense(1, 1, []float64{-p.Kd * invTf}),
			mat.NewDense(1, 1, []float64{p.Kp + p.Kd*invTf}),
			0,
		)

	default:
		invTf := 1.0 / p.Tf
		return New(
			mat.NewDense(2, 2, []float64{0, 0, 0, -invTf}),
			mat.NewDense(2, 1, []float64{1, invTf}),
			mat.NewDense(1, 2, []float64{p.Ki, -p.Kd * invTf}),
			mat.NewDense(1, 1, []float64{p.Kp + p.Kd*invTf}),
			0,
		)
	}
}

func (p *PID) discreteSystem() (*System, error) {
	hasI := p.Ki != 0
	hasD := p.Kd != 0 && p.Tf != 0
	dt := p.Dt

	switch {
	case !hasI && !hasD:
		return NewGain(mat.NewDense(1, 1, []float64{p.Kp}), dt)

	case hasI && !hasD:
		return New(
			mat.NewDense(1, 1, []float64{1}),
			mat.NewDense(1, 1, []float64{dt}),
			mat.NewDense(1, 1, []float64{p.Ki}),
			mat.NewDense(1, 1, []float64{p.Kp}),
			dt,
		)

	case !hasI && hasD:
		alpha := 1.0 - dt/p.Tf
		invTf := 1.0 / p.Tf
		return New(
			mat.NewDense(1, 1, []float64{alpha}),
			mat.NewDense(1, 1, []float64{dt * invTf}),
			mat.NewDense(1, 1, []float64{-p.Kd * invTf}),
			mat.NewDense(1, 1, []float64{p.Kp + p.Kd*invTf}),
			dt,
		)

	default:
		alpha := 1.0 - dt/p.Tf
		invTf := 1.0 / p.Tf
		return New(
			mat.NewDense(2, 2, []float64{1, 0, 0, alpha}),
			mat.NewDense(2, 1, []float64{dt, dt * invTf}),
			mat.NewDense(1, 2, []float64{p.Ki, -p.Kd * invTf}),
			mat.NewDense(1, 1, []float64{p.Kp + p.Kd*invTf}),
			dt,
		)
	}
}
