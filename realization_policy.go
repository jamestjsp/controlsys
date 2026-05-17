package controlsys

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type realizationTransformPolicy struct {
	sys     *System
	n, m, p int
}

func newRealizationTransformPolicy(sys *System) realizationTransformPolicy {
	n, m, p := sys.Dims()
	return realizationTransformPolicy{sys: sys, n: n, m: m, p: p}
}

func (p realizationTransformPolicy) requireStandard(context string) error {
	return newDescriptorPolicy(p.sys).requireStandard(context)
}

func (p realizationTransformPolicy) requireDelayFree(context string) error {
	if p.sys.HasDelay() {
		return fmt.Errorf("controlsys: %s does not support delayed systems; use Pade/AbsorbDelay first", context)
	}
	return nil
}

func (p realizationTransformPolicy) zeroOrderCopy() *System {
	return p.sys.Copy()
}

func (p realizationTransformPolicy) result(A, B, C, D *mat.Dense) (*System, error) {
	result, err := newNoCopy(A, B, C, D, p.sys.Dt)
	if err != nil {
		return nil, err
	}
	propagateIONames(result, p.sys)
	return result, nil
}
