package controlsys

type delayConversionPolicy struct {
	dt          float64
	thiranOrder int
	padeOrder   int
}

func newDelayConversionPolicy(dt float64, thiranOrder, padeOrder int) delayConversionPolicy {
	return delayConversionPolicy{dt: dt, thiranOrder: thiranOrder, padeOrder: padeOrder}
}

func (p delayConversionPolicy) applyDiscreteExternal(disc *System, contInputDelay, contOutputDelay []float64) (*System, error) {
	var err error
	disc.InputDelay, err = convertSliceDelayToDiscrete(contInputDelay, p.dt, p.thiranOrder)
	if err != nil {
		return nil, err
	}
	disc.OutputDelay, err = convertSliceDelayToDiscrete(contOutputDelay, p.dt, p.thiranOrder)
	if err != nil {
		return nil, err
	}
	if p.thiranOrder > 0 {
		disc, err = absorbFractionalDelays(disc, contInputDelay, contOutputDelay, p.dt, p.thiranOrder)
		if err != nil {
			return nil, err
		}
	}
	return disc, nil
}

func (p delayConversionPolicy) replaceDiscreteExternal(sys *System, context string) (*System, error) {
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

	inputDelay, outputDelay, err := newDelayTopology(cur).decomposableExternal(context)
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
		bank, err := buildDiscreteSampleDelayBank(inputDelay, len(inputDelay), cur.Dt, p.thiranOrder)
		if err != nil {
			return nil, err
		}
		result, err = Series(bank, result)
		if err != nil {
			return nil, err
		}
	}
	if delaySliceHasNonzero(outputDelay) {
		bank, err := buildDiscreteSampleDelayBank(outputDelay, len(outputDelay), cur.Dt, p.thiranOrder)
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

func (p delayConversionPolicy) replaceContinuousExternal(sys *System, context string) (*System, error) {
	if !sys.HasDelay() {
		return sys.Copy(), nil
	}

	cur := sys.Copy()
	if cur.Delay != nil {
		decomp := decomposedDelayMatrix(cur.Delay)
		if decomp.hasResidual() {
			return nil, &delayTopologyResidualError{context: context}
		}
		cur.Delay = nil
		cur.InputDelay = mergeDelays(cur.InputDelay, decomp.inputDelay)
		cur.OutputDelay = mergeDelays(cur.OutputDelay, decomp.outputDelay)
	}

	result := &System{A: cur.A, B: cur.B, C: cur.C, D: cur.D, Dt: cur.Dt}
	if delaySliceHasNonzero(cur.InputDelay) {
		bank, err := buildContinuousPadeDelayBank(cur.InputDelay, p.padeOrder)
		if err != nil {
			return nil, err
		}
		if bank != nil {
			result, err = Series(bank, result)
			if err != nil {
				return nil, err
			}
		}
	}

	if delaySliceHasNonzero(cur.OutputDelay) {
		bank, err := buildContinuousPadeDelayBank(cur.OutputDelay, p.padeOrder)
		if err != nil {
			return nil, err
		}
		if bank != nil {
			result, err = Series(result, bank)
			if err != nil {
				return nil, err
			}
		}
	}

	return result, nil
}
