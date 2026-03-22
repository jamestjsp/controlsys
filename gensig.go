package controlsys

import (
	"fmt"
	"math"
)

func GenSig(sigType string, period, dt float64) (t, u []float64, err error) {
	if period <= 0 {
		return nil, nil, fmt.Errorf("GenSig: period must be positive")
	}
	if dt <= 0 {
		return nil, nil, fmt.Errorf("GenSig: dt must be positive")
	}

	steps := int(period/dt) + 1
	t = make([]float64, steps)
	u = make([]float64, steps)

	for k := range t {
		t[k] = float64(k) * dt
	}

	switch sigType {
	case "step":
		for k := range u {
			u[k] = 1
		}
	case "sine":
		for k := range u {
			u[k] = math.Sin(2 * math.Pi * t[k] / period)
		}
	case "square":
		for k := range u {
			v := math.Sin(2 * math.Pi * t[k] / period)
			if v >= 0 {
				u[k] = 1
			} else {
				u[k] = -1
			}
		}
	case "pulse":
		u[0] = 1.0 / dt
	default:
		return nil, nil, fmt.Errorf("GenSig: unknown signal type %q (use step, sine, square, pulse)", sigType)
	}

	return t, u, nil
}
