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

// WithThiranOrder enables Thiran allpass modeling for fractional discrete
// delays in SafeFeedback. Exact integer discrete delays remain state-space
// delays; continuous-time delays use Pade approximation.
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
	return newDelayConversionPolicy(sys.Dt, thiranOrder, 0).replaceDiscreteExternal(sys, "SafeFeedback")
}

func replaceContinuousDelays(sys *System, padeOrder int) (*System, error) {
	return newDelayConversionPolicy(sys.Dt, 0, padeOrder).replaceContinuousExternal(sys, "SafeFeedback")
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
