package controlsys

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type SafeFeedbackOption func(*safeFeedbackConfig)

type safeFeedbackConfig struct {
	padeOrder   int
	thiranOrder int
}

func WithPadeOrder(n int) SafeFeedbackOption {
	return func(c *safeFeedbackConfig) {
		c.padeOrder = n
	}
}

// WithThiranOrder configures SafeFeedback to replace discrete-time fractional
// external delays with Thiran allpass delay models before closing the loop.
// Integer discrete delays remain exact delay states.
func WithThiranOrder(n int) SafeFeedbackOption {
	return func(c *safeFeedbackConfig) {
		c.thiranOrder = n
	}
}

func SafeFeedback(plant, controller *System, sign float64, opts ...SafeFeedbackOption) (*System, error) {
	if err := domainMatch(plant, controller); err != nil {
		return nil, err
	}

	cfg := safeFeedbackConfig{padeOrder: 5}
	for _, o := range opts {
		o(&cfg)
	}

	strategy := safeFeedbackDelayStrategy{cfg: cfg}
	p, c, err := strategy.prepare(plant, controller, sign)
	if err != nil {
		return nil, err
	}

	result, err := Feedback(p, c, sign)
	if err != nil {
		return nil, err
	}
	result.InputName = copyStringSlice(plant.InputName)
	result.OutputName = copyStringSlice(plant.OutputName)
	return result, nil
}

type safeFeedbackDelayStrategy struct {
	cfg safeFeedbackConfig
}

func (s safeFeedbackDelayStrategy) prepare(plant, controller *System, sign float64) (*System, *System, error) {
	if plant.IsDiscrete() {
		p, err := s.replaceDiscreteDelays(plant, "plant")
		if err != nil {
			return nil, nil, err
		}
		c, err := s.replaceDiscreteDelays(controller, "controller")
		if err != nil {
			return nil, nil, err
		}
		return p, c, nil
	}

	p, err := replaceContinuousDelays(plant, s.cfg.padeOrder)
	if err != nil {
		return nil, nil, fmt.Errorf("SafeFeedback: pade plant: %w", err)
	}
	c, err := replaceContinuousDelays(controller, s.cfg.padeOrder)
	if err != nil {
		return nil, nil, fmt.Errorf("SafeFeedback: pade controller: %w", err)
	}
	if err := s.requireWellPosedPade(p, c, sign); err != nil {
		return nil, nil, err
	}
	return p, c, nil
}

func (s safeFeedbackDelayStrategy) replaceDiscreteDelays(sys *System, role string) (*System, error) {
	if s.cfg.thiranOrder == 0 {
		if err := sys.Validate(); err != nil {
			return nil, fmt.Errorf("SafeFeedback: validate %s: %w", role, err)
		}
		out, err := sys.AbsorbDelay()
		if err != nil {
			return nil, fmt.Errorf("SafeFeedback: absorb %s: %w", role, err)
		}
		return out, nil
	}
	out, err := replaceDiscreteExternalDelaysWithThiran(sys, s.cfg.thiranOrder)
	if err != nil {
		return nil, fmt.Errorf("SafeFeedback: thiran %s: %w", role, err)
	}
	return out, nil
}

func (s safeFeedbackDelayStrategy) requireWellPosedPade(plant, controller *System, sign float64) error {
	_, mPlant, pPlant := plant.Dims()
	_, mCtrl, pCtrl := controller.Dims()
	if pPlant != mCtrl || pCtrl != mPlant {
		return nil
	}
	if _, err := solveFeedbackFeedthrough(plant.D, controller.D, sign, pPlant, "SafeFeedback", ErrAlgebraicLoop); err != nil {
		return fmt.Errorf("SafeFeedback: Pade approximation creates singular algebraic loop; try a different padeOrder (even vs odd) to flip feedthrough sign")
	}
	return nil
}

func replaceDiscreteExternalDelaysWithThiran(sys *System, thiranOrder int) (*System, error) {
	if !sys.HasDelay() {
		return sys.Copy(), nil
	}
	if sys.HasInternalDelay() {
		if err := validateSliceDelay(sys.LFT.Tau, len(sys.LFT.Tau), sys.Dt); err != nil {
			return nil, err
		}
	}

	cur := sys.Copy()
	if cur.HasInternalDelay() {
		var err error
		cur, err = absorbInternalDelay(cur)
		if err != nil {
			return nil, err
		}
	}

	inputDelay, outputDelay, err := newDelayTopology(cur).decomposableExternal("SafeFeedback")
	if err != nil {
		return nil, err
	}
	if !delaySliceHasNonzero(inputDelay) && !delaySliceHasNonzero(outputDelay) {
		cur.Delay = nil
		cur.InputDelay = nil
		cur.OutputDelay = nil
		return cur, nil
	}

	result := &System{A: cur.A, B: cur.B, C: cur.C, D: cur.D, Dt: cur.Dt}
	if delaySliceHasNonzero(inputDelay) {
		bank, err := buildDiscreteSampleDelayBank(inputDelay, len(inputDelay), cur.Dt, thiranOrder)
		if err != nil {
			return nil, err
		}
		result, err = Series(bank, result)
		if err != nil {
			return nil, err
		}
	}
	if delaySliceHasNonzero(outputDelay) {
		bank, err := buildDiscreteSampleDelayBank(outputDelay, len(outputDelay), cur.Dt, thiranOrder)
		if err != nil {
			return nil, err
		}
		result, err = Series(result, bank)
		if err != nil {
			return nil, err
		}
	}
	propagateNames(result, cur)
	result.InputDelay = nil
	result.OutputDelay = nil
	result.Delay = nil
	return result, nil
}

func replaceContinuousDelays(sys *System, padeOrder int) (*System, error) {
	if !sys.HasDelay() {
		return sys.Copy(), nil
	}

	_, m, p := sys.Dims()

	cur := sys.Copy()

	if cur.Delay != nil {
		decomp := decomposedDelayMatrix(cur.Delay)
		if decomp.hasResidual() {
			return nil, &delayTopologyResidualError{context: "SafeFeedback"}
		}
		cur.Delay = nil
		cur.InputDelay = mergeDelays(cur.InputDelay, decomp.inputDelay)
		cur.OutputDelay = mergeDelays(cur.OutputDelay, decomp.outputDelay)
	}

	result := &System{A: cur.A, B: cur.B, C: cur.C, D: cur.D, Dt: cur.Dt}

	if cur.InputDelay != nil {
		for j := m - 1; j >= 0; j-- {
			tau := cur.InputDelay[j]
			if tau == 0 {
				continue
			}
			pade, err := PadeDelay(tau, padeOrder)
			if err != nil {
				return nil, err
			}
			diag, err := buildDiagWithPade(j, m, pade)
			if err != nil {
				return nil, err
			}
			result, err = Series(diag, result)
			if err != nil {
				return nil, err
			}
		}
	}

	if cur.OutputDelay != nil {
		for i := p - 1; i >= 0; i-- {
			tau := cur.OutputDelay[i]
			if tau == 0 {
				continue
			}
			pade, err := PadeDelay(tau, padeOrder)
			if err != nil {
				return nil, err
			}
			diag, err := buildDiagWithPade(i, p, pade)
			if err != nil {
				return nil, err
			}
			result, err = Series(result, diag)
			if err != nil {
				return nil, err
			}
		}
	}

	return result, nil
}

func buildDiagWithPade(channel, size int, pade *System) (*System, error) {
	if size == 1 {
		return pade, nil
	}

	np, _, _ := pade.Dims()

	n := np
	A := mat.NewDense(n, n, nil)
	B := mat.NewDense(n, size, nil)
	C := mat.NewDense(size, n, nil)
	dData := make([]float64, size*size)
	for i := 0; i < size; i++ {
		dData[i*size+i] = 1
	}
	D := mat.NewDense(size, size, dData)

	if np > 0 {
		setBlock(A, 0, 0, pade.A)

		padeB := pade.B.RawMatrix()
		for i := 0; i < np; i++ {
			B.Set(i, channel, padeB.Data[i*padeB.Stride])
		}

		padeC := pade.C.RawMatrix()
		for j := 0; j < np; j++ {
			C.Set(channel, j, padeC.Data[j])
		}
	}

	padeD := pade.D.At(0, 0)
	D.Set(channel, channel, padeD)

	return newNoCopy(A, B, C, D, pade.Dt)
}
