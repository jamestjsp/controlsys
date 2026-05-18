package controlsys

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type delayBankSpec struct {
	sampleDelay             []float64
	channels                int
	dt                      float64
	thiranOrder             int
	requiresFractionalModel bool
}

func buildContinuousDelayBank(contDelay []float64, channels int, dt float64, thiranOrder int) (*System, error) {
	sampleDelay := make([]float64, channels)
	for i := 0; i < channels; i++ {
		sampleDelay[i] = contDelay[i] / dt
	}
	return buildDelayBank(delayBankSpec{
		sampleDelay:             sampleDelay,
		channels:                channels,
		dt:                      dt,
		thiranOrder:             thiranOrder,
		requiresFractionalModel: true,
	})
}

func buildDiscreteSampleDelayBank(sampleDelay []float64, channels int, dt float64, thiranOrder int) (*System, error) {
	return buildDelayBank(delayBankSpec{
		sampleDelay: sampleDelay,
		channels:    channels,
		dt:          dt,
		thiranOrder: thiranOrder,
	})
}

func buildDelayBank(spec delayBankSpec) (*System, error) {
	if spec.requiresFractionalModel && !hasFractionalSampleDelay(spec.sampleDelay) {
		return nil, nil
	}

	var bank *System
	for i := 0; i < spec.channels; i++ {
		ch, err := buildDelayChannel(spec.sampleDelay[i], spec.dt, spec.thiranOrder)
		if err != nil {
			return nil, err
		}
		if bank == nil {
			bank = ch
			continue
		}
		bank, err = Append(bank, ch)
		if err != nil {
			return nil, err
		}
	}
	return bank, nil
}

func hasFractionalSampleDelay(sampleDelay []float64) bool {
	for _, samples := range sampleDelay {
		if samples == 0 {
			continue
		}
		if !isIntegerSampleDelay(samples) {
			return true
		}
	}
	return false
}

func buildDelayChannel(samples, dt float64, thiranOrder int) (*System, error) {
	if samples == 0 {
		return NewGain(mat.NewDense(1, 1, []float64{1}), dt)
	}
	if isIntegerSampleDelay(samples) {
		return integerDelaySS(int(math.Round(samples)), dt)
	}
	return ThiranDelay(samples*dt, thiranOrder, dt)
}

func isIntegerSampleDelay(samples float64) bool {
	return math.Abs(samples-math.Round(samples)) < 1e-9
}
