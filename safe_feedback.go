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

	var p, c *System
	var err error

	if plant.IsDiscrete() {
		p, err = plant.AbsorbDelay()
		if err != nil {
			return nil, fmt.Errorf("SafeFeedback: absorb plant: %w", err)
		}
		c, err = controller.AbsorbDelay()
		if err != nil {
			return nil, fmt.Errorf("SafeFeedback: absorb controller: %w", err)
		}
	} else {
		p, err = replaceContinuousDelays(plant, cfg.padeOrder)
		if err != nil {
			return nil, fmt.Errorf("SafeFeedback: pade plant: %w", err)
		}
		c, err = replaceContinuousDelays(controller, cfg.padeOrder)
		if err != nil {
			return nil, fmt.Errorf("SafeFeedback: pade controller: %w", err)
		}

		// Well-posedness check for Pade approximations (§7.6)
		// Pade approximants have D = (-1)^N, which might create a singular algebraic loop.
		_, mPlant, pPlant := p.Dims()
		_, mCtrl, pCtrl := c.Dims()
		if pPlant == mCtrl && pCtrl == mPlant {
			dp := p.D
			dc := c.D
			tmp := mat.NewDense(pPlant, pPlant, nil)
			tmp.Product(dp, dc)
			tmp.Scale(sign, tmp)
			eye := mat.NewDense(pPlant, pPlant, nil)
			for i := 0; i < pPlant; i++ {
				eye.Set(i, i, 1)
			}
			eye.Sub(eye, tmp)
			if det := mat.Det(eye); det == 0 {
				return nil, fmt.Errorf("SafeFeedback: Pade approximation creates singular algebraic loop; try a different padeOrder (even vs odd) to flip feedthrough sign")
			}
		}
	}

	return Feedback(p, c, sign)
}

func replaceContinuousDelays(sys *System, padeOrder int) (*System, error) {
	if !sys.HasDelay() {
		return sys.Copy(), nil
	}

	_, m, p := sys.Dims()

	cur := sys.Copy()

	if cur.Delay != nil {
		inDel, outDel, residual := DecomposeIODelay(cur.Delay)
		if cur.InputDelay == nil {
			cur.InputDelay = make([]float64, m)
		}
		for j := 0; j < m; j++ {
			cur.InputDelay[j] += inDel[j]
		}
		if cur.OutputDelay == nil {
			cur.OutputDelay = make([]float64, p)
		}
		for i := 0; i < p; i++ {
			cur.OutputDelay[i] += outDel[i]
		}

		hasResidual := false
		if residual != nil {
			raw := residual.RawMatrix()
			for i := 0; i < raw.Rows; i++ {
				for j := 0; j < raw.Cols; j++ {
					if raw.Data[i*raw.Stride+j] != 0 {
						hasResidual = true
						break
					}
				}
				if hasResidual {
					break
				}
			}
		}
		if hasResidual {
			return nil, fmt.Errorf("SafeFeedback: non-decomposable IODelay residual: %w", ErrFeedbackDelay)
		}
		cur.Delay = nil
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
