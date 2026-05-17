package controlsys

import (
	"fmt"
	"math"
)

type c2dPlan struct {
	sys             *System
	dt              float64
	opts            C2DOptions
	method          string
	delayModeling   string
	contInputDelay  []float64
	contOutputDelay []float64
	workSys         *System
}

func newC2DPlan(sys *System, dt float64, opts C2DOptions) (c2dPlan, error) {
	if sys.IsDiscrete() {
		return c2dPlan{}, fmt.Errorf("DiscretizeWithOpts: system already discrete: %w", ErrWrongDomain)
	}
	if dt <= 0 {
		return c2dPlan{}, ErrInvalidSampleTime
	}

	method := opts.Method
	if method == "" {
		method = "zoh"
	}
	switch method {
	case "zoh", "tustin", "foh", "impulse", "matched":
	default:
		return c2dPlan{}, fmt.Errorf("DiscretizeWithOpts: unknown method %q", method)
	}
	opts.Method = method

	delayModeling := opts.DelayModeling
	if delayModeling == "" {
		delayModeling = "state"
	}
	switch delayModeling {
	case "state", "internal":
	default:
		return c2dPlan{}, fmt.Errorf("DiscretizeWithOpts: unknown DelayModeling %q", delayModeling)
	}

	cp := sys.Copy()
	plan := c2dPlan{
		sys:             sys,
		dt:              dt,
		opts:            opts,
		method:          method,
		delayModeling:   delayModeling,
		contInputDelay:  sys.InputDelay,
		contOutputDelay: sys.OutputDelay,
		workSys:         cp,
	}
	plan.workSys.InputDelay = nil
	plan.workSys.OutputDelay = nil
	if sys.Delay != nil && (opts.ThiranOrder > 0 || delayModeling == "internal") {
		decomp := decomposedDelayMatrix(sys.Delay)
		if decomp.hasResidual() {
			plan.workSys.Delay = decomp.residual
		} else {
			plan.workSys.Delay = nil
		}
		plan.contInputDelay = mergeDelays(sys.InputDelay, decomp.inputDelay)
		plan.contOutputDelay = mergeDelays(sys.OutputDelay, decomp.outputDelay)
	}

	return plan, nil
}

func (p c2dPlan) run() (*System, error) {
	if p.workSys.HasInternalDelay() {
		return p.discretizeInternalDelay()
	}

	disc, err := p.discretizeMethod()
	if err != nil {
		return nil, err
	}

	if p.delayModeling == "internal" {
		return discretizeDelaysAsInternal(disc, p.contInputDelay, p.contOutputDelay, p.dt)
	}
	return p.applyExternalDelays(disc)
}

func (p c2dPlan) discretizeInternalDelay() (*System, error) {
	disc, err := discretizeWithInternalDelay(p.workSys, p.dt, p.opts)
	if err != nil {
		return nil, err
	}
	return p.applyExternalDelays(disc)
}

func (p c2dPlan) discretizeMethod() (*System, error) {
	switch p.method {
	case "zoh":
		return p.workSys.DiscretizeZOH(p.dt)
	case "tustin":
		return p.workSys.Discretize(p.dt)
	case "foh":
		return p.workSys.DiscretizeFOH(p.dt)
	case "impulse":
		return p.workSys.DiscretizeImpulse(p.dt)
	case "matched":
		return p.workSys.DiscretizeMatched(p.dt)
	default:
		panic("unvalidated C2D method")
	}
}

func (p c2dPlan) applyExternalDelays(disc *System) (*System, error) {
	var err error
	disc.InputDelay, err = convertSliceDelayToDiscrete(p.contInputDelay, p.dt, p.opts.ThiranOrder)
	if err != nil {
		return nil, err
	}
	disc.OutputDelay, err = convertSliceDelayToDiscrete(p.contOutputDelay, p.dt, p.opts.ThiranOrder)
	if err != nil {
		return nil, err
	}
	if p.opts.ThiranOrder > 0 {
		disc, err = absorbFractionalDelays(disc, p.contInputDelay, p.contOutputDelay, p.dt, p.opts.ThiranOrder)
		if err != nil {
			return nil, err
		}
	}
	return disc, nil
}

type d2cPlan struct {
	sys    *System
	method string
}

func newD2CPlan(sys *System, method string) (d2cPlan, error) {
	if sys.IsContinuous() {
		return d2cPlan{}, fmt.Errorf("D2C: system already continuous: %w", ErrWrongDomain)
	}
	if method == "" {
		method = "zoh"
	}
	switch method {
	case "zoh", "tustin":
		return d2cPlan{sys: sys, method: method}, nil
	default:
		return d2cPlan{}, fmt.Errorf("D2C: unknown method %q (supported: \"tustin\", \"zoh\")", method)
	}
}

func (p d2cPlan) run() (*System, error) {
	switch p.method {
	case "zoh":
		return p.sys.d2cZOH()
	case "tustin":
		return p.sys.Undiscretize()
	default:
		panic("unvalidated D2C method")
	}
}

type d2dPlan struct {
	sys   *System
	newDt float64
	opts  C2DOptions
}

func newD2DPlan(sys *System, newDt float64, opts C2DOptions) (d2dPlan, error) {
	if sys.IsContinuous() {
		return d2dPlan{}, fmt.Errorf("D2D: system is continuous: %w", ErrWrongDomain)
	}
	if newDt <= 0 {
		return d2dPlan{}, ErrInvalidSampleTime
	}
	return d2dPlan{sys: sys, newDt: newDt, opts: opts}, nil
}

func (p d2dPlan) run() (*System, error) {
	if math.Abs(p.newDt-p.sys.Dt) < 1e-14*math.Max(p.newDt, p.sys.Dt) {
		return p.sys.Copy(), nil
	}

	contSys, err := p.sys.Undiscretize()
	if err != nil {
		return nil, fmt.Errorf("D2D: %w", err)
	}
	result, err := contSys.DiscretizeWithOpts(p.newDt, p.opts)
	if err != nil {
		return nil, fmt.Errorf("D2D: %w", err)
	}
	propagateNames(result, p.sys)
	return result, nil
}
