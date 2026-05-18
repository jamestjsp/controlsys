package controlsys

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

type delayBankMode int

const (
	delayBankPade delayBankMode = iota
	delayBankDiscreteSamples
	delayBankContinuousThiran
)

type delayBankSpec struct {
	mode  delayBankMode
	order int
	dt    float64
}

func buildPadeBank(delays []float64, order int) (*System, error) {
	return buildChannelDelayBank(delays, len(delays), delayBankSpec{mode: delayBankPade, order: order})
}

func buildDiscreteSampleDelayBank(sampleDelay []float64, n int, dt float64, thiranOrder int) (*System, error) {
	return buildChannelDelayBank(sampleDelay, n, delayBankSpec{mode: delayBankDiscreteSamples, dt: dt, order: thiranOrder})
}

func buildFractionalThiranDelayBank(contDelay []float64, n int, dt float64, thiranOrder int) (*System, error) {
	if !hasFractionalSampleDelay(contDelay, dt) {
		return nil, nil
	}
	return buildChannelDelayBank(contDelay, n, delayBankSpec{mode: delayBankContinuousThiran, dt: dt, order: thiranOrder})
}

func buildChannelDelayBank(delays []float64, n int, spec delayBankSpec) (*System, error) {
	var bank *System
	for i := 0; i < n; i++ {
		ch, err := buildDelayChannel(delays[i], spec)
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

func buildDelayChannel(delay float64, spec delayBankSpec) (*System, error) {
	switch spec.mode {
	case delayBankPade:
		if delay == 0 {
			return NewGain(mat.NewDense(1, 1, []float64{1}), 0)
		}
		pd, err := PadeDelay(delay, spec.order)
		if err != nil {
			return nil, fmt.Errorf("buildPadeBank: %w", err)
		}
		return pd, nil
	case delayBankDiscreteSamples:
		return buildDiscreteSampleDelayChannel(delay, spec.dt, spec.order)
	case delayBankContinuousThiran:
		return buildContinuousThiranDelayChannel(delay, spec.dt, spec.order)
	default:
		panic("unvalidated delay bank mode")
	}
}

func buildDiscreteSampleDelayChannel(samples, dt float64, thiranOrder int) (*System, error) {
	if samples == 0 {
		return NewGain(mat.NewDense(1, 1, []float64{1}), dt)
	}
	if math.Abs(samples-math.Round(samples)) < 1e-9 {
		return integerDelaySS(int(math.Round(samples)), dt)
	}
	return ThiranDelay(samples*dt, thiranOrder, dt)
}

func buildContinuousThiranDelayChannel(tau, dt float64, thiranOrder int) (*System, error) {
	if tau == 0 {
		return NewGain(mat.NewDense(1, 1, []float64{1}), dt)
	}
	samples := tau / dt
	if math.Abs(samples-math.Round(samples)) < 1e-9 {
		return integerDelaySS(int(math.Round(samples)), dt)
	}
	return ThiranDelay(tau, thiranOrder, dt)
}

func hasFractionalSampleDelay(delays []float64, dt float64) bool {
	for _, tau := range delays {
		if tau == 0 {
			continue
		}
		samples := tau / dt
		if math.Abs(samples-math.Round(samples)) >= 1e-9 {
			return true
		}
	}
	return false
}
