package controlsys

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type interconnectionTopology struct {
	plan interconnectionPlan
}

func newInterconnectionTopology(sys1, sys2 *System) interconnectionTopology {
	return interconnectionTopology{plan: newInterconnectionPlan(sys1, sys2)}
}

func (t interconnectionTopology) seriesRequiresLFT() bool {
	plan := t.plan
	if plan.sys1.HasInternalDelay() || plan.sys2.HasInternalDelay() {
		return true
	}
	if !plan.sys1.HasDelay() && !plan.sys2.HasDelay() {
		return false
	}
	ioCheck := seriesDelay(plan.sys1, plan.sys2)
	if ioCheck == nil && (plan.sys1.Delay != nil || plan.sys2.Delay != nil) {
		return true
	}
	if plan.sys1.OutputDelay == nil && plan.sys2.InputDelay == nil {
		return false
	}
	for k := 1; k < plan.p1; k++ {
		if math.Abs(t.seriesIntermediateDelay(k)-t.seriesIntermediateDelay(0)) > delayTopologyTol {
			return true
		}
	}
	return false
}

func (t interconnectionTopology) seriesIntermediateDelay(k int) float64 {
	plan := t.plan
	var delay float64
	if plan.sys1.OutputDelay != nil {
		delay += plan.sys1.OutputDelay[k]
	}
	if plan.sys2.InputDelay != nil {
		delay += plan.sys2.InputDelay[k]
	}
	return delay
}

func (t interconnectionTopology) parallelRequiresLFT() bool {
	plan := t.plan
	if plan.sys1.HasInternalDelay() || plan.sys2.HasInternalDelay() {
		return true
	}
	if !plan.sys1.HasDelay() && !plan.sys2.HasDelay() {
		return false
	}
	td1 := t.totalDelayOrZero(plan.sys1)
	td2 := t.totalDelayOrZero(plan.sys2)
	if td1 == nil && td2 == nil {
		return false
	}
	if td1 == nil {
		td1 = mat.NewDense(plan.p1, plan.m1, nil)
	}
	if td2 == nil {
		td2 = mat.NewDense(plan.p1, plan.m1, nil)
	}
	return !mat.Equal(td1, td2)
}

func (t interconnectionTopology) totalDelayOrZero(sys *System) *mat.Dense {
	return newDelayTopology(sys).totalExternal(true)
}

func (t interconnectionTopology) leadingVisibleInputDelay(n int) visibleDelaySelection {
	return selectLeadingVisibleDelays(t.plan.sys1.InputDelay, n)
}

func (t interconnectionTopology) leadingVisibleOutputDelay(n int) visibleDelaySelection {
	return selectLeadingVisibleDelays(t.plan.sys1.OutputDelay, n)
}
