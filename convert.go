package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/mat"
)

func (sys *System) Discretize(dt float64) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("Discretize: system already discrete: %w", ErrWrongDomain)
	}
	if dt <= 0 {
		return nil, ErrInvalidSampleTime
	}

	if sys.HasInternalDelay() {
		return discretizeWithInternalDelay(sys, dt, C2DOptions{Method: "tustin"})
	}

	beta := 2.0 / dt
	if math.IsInf(beta, 0) {
		return nil, fmt.Errorf("Discretize: dt too small, 2/dt overflows: %w", ErrOverflow)
	}
	out, err := bilinear(sys, -beta, -1.0, 1.0, beta)
	if err != nil {
		return nil, err
	}
	out.Dt = dt
	if sys.Delay != nil {
		d, err := convertDelayToDiscrete(sys.Delay, dt)
		if err != nil {
			return nil, err
		}
		out.Delay = d
	}
	if sys.InputDelay != nil {
		id, err := convertSliceDelayToDiscrete(sys.InputDelay, dt, 0)
		if err != nil {
			return nil, err
		}
		out.InputDelay = id
	}
	if sys.OutputDelay != nil {
		od, err := convertSliceDelayToDiscrete(sys.OutputDelay, dt, 0)
		if err != nil {
			return nil, err
		}
		out.OutputDelay = od
	}
	propagateNames(out, sys)
	return out, nil
}

func (sys *System) Undiscretize() (*System, error) {
	if sys.IsContinuous() {
		return nil, fmt.Errorf("Undiscretize: system already continuous: %w", ErrWrongDomain)
	}
	beta := 2.0 / sys.Dt
	if math.IsInf(beta, 0) {
		return nil, fmt.Errorf("Undiscretize: dt too small, 2/dt overflows: %w", ErrOverflow)
	}
	out, err := bilinear(sys, 1.0, beta, 1.0, beta)
	if err != nil {
		return nil, err
	}
	out.Dt = 0
	if sys.Delay != nil {
		out.Delay = convertDelayToContinuous(sys.Delay, sys.Dt)
	}
	if sys.InputDelay != nil {
		out.InputDelay = make([]float64, len(sys.InputDelay))
		for i, d := range sys.InputDelay {
			out.InputDelay[i] = d * sys.Dt
		}
	}
	if sys.OutputDelay != nil {
		out.OutputDelay = make([]float64, len(sys.OutputDelay))
		for i, d := range sys.OutputDelay {
			out.OutputDelay[i] = d * sys.Dt
		}
	}
	if sys.InternalDelay != nil {
		out.InternalDelay = make([]float64, len(sys.InternalDelay))
		for i, d := range sys.InternalDelay {
			out.InternalDelay[i] = d * sys.Dt
		}
		out.B2 = mat.DenseCopyOf(sys.B2)
		out.C2 = mat.DenseCopyOf(sys.C2)
		out.D12 = mat.DenseCopyOf(sys.D12)
		out.D21 = mat.DenseCopyOf(sys.D21)
		out.D22 = mat.DenseCopyOf(sys.D22)
	}
	propagateNames(out, sys)
	return out, nil
}

func bilinear(sys *System, palpha, pbeta, alpha, beta float64) (*System, error) {
	n, m, p := sys.Dims()

	D := denseCopy(sys.D)

	if n == 0 {
		out := &System{
			A: newDense(0, 0),
			B: denseCopy(sys.B),
			C: denseCopy(sys.C),
			D: D,
		}
		propagateNames(out, sys)
		return out, nil
	}

	A := mat.NewDense(n, n, nil)
	A.Copy(sys.A)

	B := denseCopy(sys.B)
	C := denseCopy(sys.C)

	aRaw := A.RawMatrix()
	for i := 0; i < n; i++ {
		aRaw.Data[i*aRaw.Stride+i] += palpha
	}

	var lu mat.LU
	lu.Factorize(A)
	if lu.Det() == 0 {
		return nil, fmt.Errorf("bilinear: (palpha·I + A) is singular: %w", ErrSingularTransform)
	}

	if m > 0 {
		err := lu.SolveTo(B, false, B)
		if err != nil {
			return nil, fmt.Errorf("bilinear: LU solve for B failed: %w", ErrSingularTransform)
		}
	}

	if p > 0 && m > 0 {
		D.Mul(C, B)
		D.Scale(-1, D)
		D.Add(D, sys.D)
	}

	twoAB := 2.0 * alpha * beta
	if math.IsInf(twoAB, 0) {
		return nil, fmt.Errorf("bilinear: 2*alpha*beta overflows: %w", ErrOverflow)
	}
	scale := math.Sqrt(math.Abs(twoAB))
	if palpha < 0 {
		scale = -scale
	}

	if m > 0 {
		B.Scale(scale, B)
	}

	Ainv := mat.NewDense(n, n, nil)
	eye := mat.NewDense(n, n, nil)
	eyeRaw := eye.RawMatrix()
	for i := 0; i < n; i++ {
		eyeRaw.Data[i*eyeRaw.Stride+i] = 1
	}
	err := lu.SolveTo(Ainv, false, eye)
	if err != nil {
		return nil, fmt.Errorf("bilinear: LU inverse failed: %w", ErrSingularTransform)
	}

	if p > 0 {
		C.Mul(C, Ainv)
		C.Scale(scale, C)
	}
	Ainv.Scale(-twoAB, Ainv)
	ainvRaw := Ainv.RawMatrix()
	for i := 0; i < n; i++ {
		ainvRaw.Data[i*ainvRaw.Stride+i] += pbeta
	}

	if math.IsInf(denseNorm(Ainv), 0) || math.IsNaN(denseNorm(Ainv)) {
		return nil, fmt.Errorf("bilinear: result contains Inf/NaN: %w", ErrOverflow)
	}

	out := &System{A: Ainv, B: B, C: C, D: D}
	propagateNames(out, sys)
	return out, nil
}

type C2DOptions struct {
	Method        string
	ThiranOrder   int
	DelayModeling string
}

func (sys *System) DiscretizeWithOpts(dt float64, opts C2DOptions) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("DiscretizeWithOpts: system already discrete: %w", ErrWrongDomain)
	}
	if dt <= 0 {
		return nil, ErrInvalidSampleTime
	}

	delayModeling := opts.DelayModeling
	if delayModeling == "" {
		delayModeling = "state"
	}
	if delayModeling != "state" && delayModeling != "internal" {
		return nil, fmt.Errorf("DiscretizeWithOpts: unknown DelayModeling %q", delayModeling)
	}

	method := opts.Method
	if method == "" {
		method = "zoh"
	}

	contInputDelay := sys.InputDelay
	contOutputDelay := sys.OutputDelay

	cp := sys.Copy()
	cp.InputDelay = nil
	cp.OutputDelay = nil

	if sys.Delay != nil && (opts.ThiranOrder > 0 || delayModeling == "internal") {
		inD, outD, residual := DecomposeIODelay(sys.Delay)
		hasResidual := false
		raw := residual.RawMatrix()
		for _, v := range raw.Data[:raw.Rows*raw.Cols] {
			if v != 0 {
				hasResidual = true
				break
			}
		}
		if hasResidual {
			cp.Delay = residual
		} else {
			cp.Delay = nil
		}
		contInputDelay = mergeDelays(sys.InputDelay, inD)
		contOutputDelay = mergeDelays(sys.OutputDelay, outD)
	}
	workSys := cp

	if workSys.HasInternalDelay() {
		disc, err := discretizeWithInternalDelay(workSys, dt, opts)
		if err != nil {
			return nil, err
		}
		disc.InputDelay, err = convertSliceDelayToDiscrete(contInputDelay, dt, opts.ThiranOrder)
		if err != nil {
			return nil, err
		}
		disc.OutputDelay, err = convertSliceDelayToDiscrete(contOutputDelay, dt, opts.ThiranOrder)
		if err != nil {
			return nil, err
		}
		if opts.ThiranOrder > 0 {
			disc, err = absorbFractionalDelays(disc, contInputDelay, contOutputDelay, dt, opts.ThiranOrder)
			if err != nil {
				return nil, err
			}
		}
		return disc, nil
	}

	var disc *System
	var err error
	switch method {
	case "zoh":
		disc, err = workSys.DiscretizeZOH(dt)
	case "tustin":
		disc, err = workSys.Discretize(dt)
	case "foh":
		disc, err = workSys.DiscretizeFOH(dt)
	case "impulse":
		disc, err = workSys.DiscretizeImpulse(dt)
	case "matched":
		disc, err = workSys.DiscretizeMatched(dt)
	default:
		return nil, fmt.Errorf("DiscretizeWithOpts: unknown method %q", method)
	}
	if err != nil {
		return nil, err
	}

	if delayModeling == "internal" {
		return discretizeDelaysAsInternal(disc, contInputDelay, contOutputDelay, dt)
	}

	disc.InputDelay, err = convertSliceDelayToDiscrete(contInputDelay, dt, opts.ThiranOrder)
	if err != nil {
		return nil, err
	}
	disc.OutputDelay, err = convertSliceDelayToDiscrete(contOutputDelay, dt, opts.ThiranOrder)
	if err != nil {
		return nil, err
	}

	if opts.ThiranOrder > 0 {
		disc, err = absorbFractionalDelays(disc, contInputDelay, contOutputDelay, dt, opts.ThiranOrder)
		if err != nil {
			return nil, err
		}
	}

	return disc, nil
}

func discretizeDelaysAsInternal(disc *System, contInputDelay, contOutputDelay []float64, dt float64) (*System, error) {
	n, m, p := disc.Dims()

	type fracEntry struct {
		isInput bool
		idx     int
		frac    float64
	}
	var fracs []fracEntry

	inputDelayDisc := make([]float64, len(contInputDelay))
	for j, tau := range contInputDelay {
		if tau == 0 {
			continue
		}
		samples := tau / dt
		fracPart := samples - math.Floor(samples)
		if fracPart < 1e-9 || fracPart > 1-1e-9 {
			inputDelayDisc[j] = math.Round(samples)
		} else {
			inputDelayDisc[j] = math.Floor(samples)
			fracs = append(fracs, fracEntry{isInput: true, idx: j, frac: fracPart})
		}
	}

	outputDelayDisc := make([]float64, len(contOutputDelay))
	for i, tau := range contOutputDelay {
		if tau == 0 {
			continue
		}
		samples := tau / dt
		fracPart := samples - math.Floor(samples)
		if fracPart < 1e-9 || fracPart > 1-1e-9 {
			outputDelayDisc[i] = math.Round(samples)
		} else {
			outputDelayDisc[i] = math.Floor(samples)
			fracs = append(fracs, fracEntry{isInput: false, idx: i, frac: fracPart})
		}
	}

	if len(contInputDelay) > 0 {
		disc.InputDelay = inputDelayDisc
	}
	if len(contOutputDelay) > 0 {
		disc.OutputDelay = outputDelayDisc
	}

	if len(fracs) == 0 {
		return disc, nil
	}

	N := len(fracs)
	internalDelay := make([]float64, N)
	b2Data := make([]float64, n*N)
	c2Data := make([]float64, N*n)
	d12Data := make([]float64, p*N)
	d21Data := make([]float64, N*m)
	d22Data := make([]float64, N*N)

	bRaw := disc.B.RawMatrix()
	cRaw := disc.C.RawMatrix()
	dRaw := disc.D.RawMatrix()

	newBData := make([]float64, n*m)
	for i := 0; i < n; i++ {
		copy(newBData[i*m:i*m+m], bRaw.Data[i*bRaw.Stride:i*bRaw.Stride+m])
	}
	newCData := make([]float64, p*n)
	for i := 0; i < p; i++ {
		copy(newCData[i*n:i*n+n], cRaw.Data[i*cRaw.Stride:i*cRaw.Stride+n])
	}
	newDData := make([]float64, p*m)
	for i := 0; i < p; i++ {
		copy(newDData[i*m:i*m+m], dRaw.Data[i*dRaw.Stride:i*dRaw.Stride+m])
	}

	for k, f := range fracs {
		internalDelay[k] = f.frac
		if f.isInput {
			j := f.idx
			d21Data[k*m+j] = 1
			for i := 0; i < n; i++ {
				b2Data[i*N+k] = newBData[i*m+j]
				newBData[i*m+j] = 0
			}
			for i := 0; i < p; i++ {
				d12Data[i*N+k] = newDData[i*m+j]
				newDData[i*m+j] = 0
			}
		} else {
			i := f.idx
			for col := 0; col < n; col++ {
				c2Data[k*n+col] = newCData[i*n+col]
				newCData[i*n+col] = 0
			}
			for col := 0; col < m; col++ {
				d21Data[k*m+col] = newDData[i*m+col]
				newDData[i*m+col] = 0
			}
			d12Data[i*N+k] = 1
		}
	}

	if n > 0 && m > 0 {
		disc.B = mat.NewDense(n, m, newBData)
	}
	if p > 0 && n > 0 {
		disc.C = mat.NewDense(p, n, newCData)
	}
	if p > 0 && m > 0 {
		disc.D = mat.NewDense(p, m, newDData)
	}

	disc.InternalDelay = internalDelay
	disc.B2 = mat.NewDense(n, N, b2Data)
	disc.C2 = mat.NewDense(N, n, c2Data)
	disc.D12 = mat.NewDense(p, N, d12Data)
	disc.D21 = mat.NewDense(N, m, d21Data)
	disc.D22 = mat.NewDense(N, N, d22Data)

	return disc, nil
}

func mergeDelays(existing, decomposed []float64) []float64 {
	if decomposed == nil {
		return existing
	}
	allZero := true
	for _, v := range decomposed {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		return existing
	}
	n := len(decomposed)
	out := make([]float64, n)
	copy(out, decomposed)
	if existing != nil {
		for i := range existing {
			if i < n {
				out[i] += existing[i]
			}
		}
	}
	return out
}

func convertSliceDelayToDiscrete(delay []float64, dt float64, thiranOrder int) ([]float64, error) {
	if delay == nil {
		return nil, nil
	}
	out := make([]float64, len(delay))
	for i, tau := range delay {
		if tau == 0 {
			continue
		}
		samples := tau / dt
		rounded := math.Round(samples)
		if math.Abs(samples-rounded) < 1e-9 {
			out[i] = rounded
		} else if thiranOrder > 0 {
			out[i] = 0
		} else {
			return nil, fmt.Errorf("delay[%d]=%g not integer multiple of dt=%g: %w",
				i, tau, dt, ErrFractionalDelay)
		}
	}
	return out, nil
}

func absorbFractionalDelays(disc *System, contInputDelay, contOutputDelay []float64, dt float64, thiranOrder int) (*System, error) {
	_, m, p := disc.Dims()

	if contInputDelay != nil {
		bank, err := buildDelayBank(contInputDelay, m, dt, thiranOrder)
		if err != nil {
			return nil, err
		}
		if bank != nil {
			disc, err = Series(bank, disc)
			if err != nil {
				return nil, err
			}
			disc.InputDelay = nil
		}
	}

	if contOutputDelay != nil {
		bank, err := buildDelayBank(contOutputDelay, p, dt, thiranOrder)
		if err != nil {
			return nil, err
		}
		if bank != nil {
			disc, err = Series(disc, bank)
			if err != nil {
				return nil, err
			}
			disc.OutputDelay = nil
		}
	}

	return disc, nil
}

func buildDelayBank(contDelay []float64, n int, dt float64, thiranOrder int) (*System, error) {
	hasFractional := false
	for _, tau := range contDelay {
		if tau == 0 {
			continue
		}
		samples := tau / dt
		if math.Abs(samples-math.Round(samples)) >= 1e-9 {
			hasFractional = true
			break
		}
	}
	if !hasFractional {
		return nil, nil
	}

	var bank *System
	for i := 0; i < n; i++ {
		tau := contDelay[i]
		var ch *System
		var err error

		if tau == 0 {
			ch, err = NewGain(mat.NewDense(1, 1, []float64{1}), dt)
			if err != nil {
				return nil, err
			}
		} else {
			samples := tau / dt
			if math.Abs(samples-math.Round(samples)) < 1e-9 {
				ch, err = integerDelaySS(int(math.Round(samples)), dt)
				if err != nil {
					return nil, err
				}
			} else {
				ch, err = ThiranDelay(tau, thiranOrder, dt)
				if err != nil {
					return nil, err
				}
			}
		}

		if bank == nil {
			bank = ch
		} else {
			bank, err = Append(bank, ch)
			if err != nil {
				return nil, err
			}
		}
	}
	return bank, nil
}

func (sys *System) DiscretizeZOH(dt float64) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("DiscretizeZOH: system already discrete: %w", ErrWrongDomain)
	}
	if dt <= 0 {
		return nil, ErrInvalidSampleTime
	}

	if sys.HasInternalDelay() {
		return discretizeWithInternalDelay(sys, dt, C2DOptions{Method: "zoh"})
	}

	n, m, _ := sys.Dims()

	D := denseCopy(sys.D)
	C := denseCopy(sys.C)

	if n == 0 {
		out := &System{
			A:  newDense(0, 0),
			B:  denseCopy(sys.B),
			C:  C,
			D:  D,
			Dt: dt,
		}
		propagateNames(out, sys)
		return out, nil
	}

	nm := n + m
	if m == 0 {
		var eA mat.Dense
		Adt := mat.NewDense(n, n, nil)
		Adt.Scale(dt, sys.A)
		eA.Exp(Adt)
		Ad := mat.NewDense(n, n, nil)
		Ad.Copy(&eA)
		out := &System{A: Ad, B: denseCopy(sys.B), C: C, D: D, Dt: dt}
		propagateNames(out, sys)
		return out, nil
	}

	M := mat.NewDense(nm, nm, nil)
	mRaw := M.RawMatrix()
	aRaw := sys.A.RawMatrix()
	bRaw := sys.B.RawMatrix()
	for i := 0; i < n; i++ {
		mRow := mRaw.Data[i*mRaw.Stride:]
		aRow := aRaw.Data[i*aRaw.Stride : i*aRaw.Stride+n]
		for j, v := range aRow {
			mRow[j] = v * dt
		}
		bRow := bRaw.Data[i*bRaw.Stride : i*bRaw.Stride+m]
		for j, v := range bRow {
			mRow[n+j] = v * dt
		}
	}

	var eM mat.Dense
	eM.Exp(M)

	emRaw := eM.RawMatrix()
	adData := make([]float64, n*n)
	bdData := make([]float64, n*m)
	for i := 0; i < n; i++ {
		copy(adData[i*n:], emRaw.Data[i*emRaw.Stride:i*emRaw.Stride+n])
		copy(bdData[i*m:], emRaw.Data[i*emRaw.Stride+n:i*emRaw.Stride+n+m])
	}
	Ad := mat.NewDense(n, n, adData)
	Bd := mat.NewDense(n, m, bdData)

	out := &System{A: Ad, B: Bd, C: C, D: D, Dt: dt}
	if sys.Delay != nil {
		d, err := convertDelayToDiscrete(sys.Delay, dt)
		if err != nil {
			return nil, err
		}
		out.Delay = d
	}
	if sys.InputDelay != nil {
		id, err := convertSliceDelayToDiscrete(sys.InputDelay, dt, 0)
		if err != nil {
			return nil, err
		}
		out.InputDelay = id
	}
	if sys.OutputDelay != nil {
		od, err := convertSliceDelayToDiscrete(sys.OutputDelay, dt, 0)
		if err != nil {
			return nil, err
		}
		out.OutputDelay = od
	}
	propagateNames(out, sys)
	return out, nil
}

func discretizeWithInternalDelay(sys *System, dt float64, opts C2DOptions) (*System, error) {
	if !isStrictlyUpperTriangular(sys.D22) {
		return nil, fmt.Errorf("DiscretizeWithOpts: ZOH not supported for GLTI with non-upper-triangular D22: %w", ErrAlgebraicLoop)
	}

	method := opts.Method
	if method == "" {
		method = "zoh"
	}

	n, _, _ := sys.Dims()
	N := len(sys.InternalDelay)

	discTau := make([]float64, N)
	for j, tau := range sys.InternalDelay {
		samples := tau / dt
		rounded := math.Round(samples)
		if math.Abs(samples-rounded) < 1e-9 {
			discTau[j] = rounded
		} else {
			return nil, fmt.Errorf("InternalDelay[%d]=%g not integer multiple of dt=%g: %w",
				j, tau, dt, ErrFractionalDelay)
		}
	}

	var disc *System
	var Bd2 *mat.Dense
	var err error

	switch method {
	case "zoh":
		disc, Bd2, err = discretizeZOHAugmented(sys, dt)
	case "tustin":
		disc, Bd2, err = discretizeTustinAugmented(sys, dt)
	case "foh":
		disc, Bd2, err = discretizeFOHAugmented(sys, dt)
	case "impulse":
		disc, Bd2, err = discretizeImpulseAugmented(sys, dt)
	default:
		return nil, fmt.Errorf("DiscretizeWithOpts: unknown method %q", method)
	}
	if err != nil {
		return nil, err
	}

	disc.InternalDelay = discTau
	if n > 0 && N > 0 {
		disc.B2 = Bd2
	} else {
		disc.B2 = mat.DenseCopyOf(sys.B2)
	}
	disc.C2 = mat.DenseCopyOf(sys.C2)
	disc.D12 = mat.DenseCopyOf(sys.D12)
	disc.D21 = mat.DenseCopyOf(sys.D21)
	disc.D22 = mat.DenseCopyOf(sys.D22)

	if sys.Delay != nil {
		d, convErr := convertDelayToDiscrete(sys.Delay, dt)
		if convErr != nil {
			return nil, convErr
		}
		disc.Delay = d
	}
	if sys.InputDelay != nil {
		id, convErr := convertSliceDelayToDiscrete(sys.InputDelay, dt, 0)
		if convErr != nil {
			return nil, convErr
		}
		disc.InputDelay = id
	}
	if sys.OutputDelay != nil {
		od, convErr := convertSliceDelayToDiscrete(sys.OutputDelay, dt, 0)
		if convErr != nil {
			return nil, convErr
		}
		disc.OutputDelay = od
	}

	propagateNames(disc, sys)
	return disc, nil
}

func discretizeZOHAugmented(sys *System, dt float64) (*System, *mat.Dense, error) {
	n, m, _ := sys.Dims()
	N := len(sys.InternalDelay)

	C := denseCopy(sys.C)
	D := denseCopy(sys.D)

	if n == 0 {
		out := &System{A: newDense(0, 0), B: denseCopy(sys.B), C: C, D: D, Dt: dt}
		return out, mat.DenseCopyOf(sys.B2), nil
	}

	mTotal := m + N
	nm := n + mTotal
	if mTotal == 0 {
		var eA mat.Dense
		Adt := mat.NewDense(n, n, nil)
		Adt.Scale(dt, sys.A)
		eA.Exp(Adt)
		Ad := mat.NewDense(n, n, nil)
		Ad.Copy(&eA)
		out := &System{A: Ad, B: denseCopy(sys.B), C: C, D: D, Dt: dt}
		return out, mat.DenseCopyOf(sys.B2), nil
	}

	M := mat.NewDense(nm, nm, nil)
	mRaw := M.RawMatrix()
	aRaw := sys.A.RawMatrix()

	for i := 0; i < n; i++ {
		row := mRaw.Data[i*mRaw.Stride:]
		for j := 0; j < n; j++ {
			row[j] = aRaw.Data[i*aRaw.Stride+j] * dt
		}
		if m > 0 {
			bRaw := sys.B.RawMatrix()
			for j := 0; j < m; j++ {
				row[n+j] = bRaw.Data[i*bRaw.Stride+j] * dt
			}
		}
		if N > 0 {
			b2Raw := sys.B2.RawMatrix()
			for j := 0; j < N; j++ {
				row[n+m+j] = b2Raw.Data[i*b2Raw.Stride+j] * dt
			}
		}
	}

	var eM mat.Dense
	eM.Exp(M)
	emRaw := eM.RawMatrix()

	adData := make([]float64, n*n)
	bdData := make([]float64, n*m)
	bd2Data := make([]float64, n*N)
	for i := 0; i < n; i++ {
		copy(adData[i*n:i*n+n], emRaw.Data[i*emRaw.Stride:i*emRaw.Stride+n])
		if m > 0 {
			copy(bdData[i*m:i*m+m], emRaw.Data[i*emRaw.Stride+n:i*emRaw.Stride+n+m])
		}
		if N > 0 {
			copy(bd2Data[i*N:i*N+N], emRaw.Data[i*emRaw.Stride+n+m:i*emRaw.Stride+n+m+N])
		}
	}

	Ad := mat.NewDense(n, n, adData)
	var Bd *mat.Dense
	if m > 0 {
		Bd = mat.NewDense(n, m, bdData)
	} else {
		Bd = newDense(n, 0)
	}
	var Bd2 *mat.Dense
	if N > 0 {
		Bd2 = mat.NewDense(n, N, bd2Data)
	} else {
		Bd2 = newDense(n, 0)
	}

	out := &System{A: Ad, B: Bd, C: C, D: D, Dt: dt}
	return out, Bd2, nil
}

func discretizeTustinAugmented(sys *System, dt float64) (*System, *mat.Dense, error) {
	n, _, _ := sys.Dims()
	N := len(sys.InternalDelay)

	beta := 2.0 / dt
	if math.IsInf(beta, 0) {
		return nil, nil, fmt.Errorf("discretizeTustinAugmented: dt too small: %w", ErrOverflow)
	}

	rational := &System{A: sys.A, B: sys.B, C: sys.C, D: sys.D}
	disc, err := bilinear(rational, -beta, -1.0, 1.0, beta)
	if err != nil {
		return nil, nil, err
	}
	disc.Dt = dt

	if n == 0 || N == 0 {
		return disc, mat.DenseCopyOf(sys.B2), nil
	}

	palpha := -beta
	A := mat.NewDense(n, n, nil)
	A.Copy(sys.A)
	aRaw := A.RawMatrix()
	for i := 0; i < n; i++ {
		aRaw.Data[i*aRaw.Stride+i] += palpha
	}

	var lu mat.LU
	lu.Factorize(A)

	B2 := mat.DenseCopyOf(sys.B2)
	if err := lu.SolveTo(B2, false, B2); err != nil {
		return nil, nil, fmt.Errorf("discretizeTustinAugmented: LU solve for B2 failed: %w", ErrSingularTransform)
	}

	twoAB := 2.0 * 1.0 * beta
	scale := math.Sqrt(math.Abs(twoAB))
	if palpha < 0 {
		scale = -scale
	}
	B2.Scale(scale, B2)

	return disc, B2, nil
}


func isStrictlyUpperTriangular(m *mat.Dense) bool {
	if m == nil {
		return true
	}
	raw := m.RawMatrix()
	for i := 0; i < raw.Rows; i++ {
		bound := i
		if raw.Cols-1 < bound {
			bound = raw.Cols - 1
		}
		row := raw.Data[i*raw.Stride:]
		for j := 0; j <= bound; j++ {
			if row[j] != 0 {
				return false
			}
		}
	}
	return true
}

func (sys *System) DiscretizeImpulse(dt float64) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("DiscretizeImpulse: system already discrete: %w", ErrWrongDomain)
	}
	if dt <= 0 {
		return nil, ErrInvalidSampleTime
	}

	if sys.HasInternalDelay() {
		return discretizeWithInternalDelay(sys, dt, C2DOptions{Method: "impulse"})
	}

	n, m, _ := sys.Dims()

	C := denseCopy(sys.C)
	D := denseCopy(sys.D)

	if n == 0 {
		out := &System{
			A:  newDense(0, 0),
			B:  denseCopy(sys.B),
			C:  C,
			D:  D,
			Dt: dt,
		}
		propagateNames(out, sys)
		return out, nil
	}

	Adt := mat.NewDense(n, n, nil)
	Adt.Scale(dt, sys.A)
	var eA mat.Dense
	eA.Exp(Adt)
	Ad := mat.NewDense(n, n, nil)
	Ad.Copy(&eA)

	if m == 0 {
		out := &System{A: Ad, B: denseCopy(sys.B), C: C, D: D, Dt: dt}
		propagateNames(out, sys)
		return out, nil
	}

	Bd := mat.NewDense(n, m, nil)
	Bd.Mul(Ad, sys.B)
	Bd.Scale(dt, Bd)

	out := &System{A: Ad, B: Bd, C: C, D: D, Dt: dt}
	if sys.Delay != nil {
		d, err := convertDelayToDiscrete(sys.Delay, dt)
		if err != nil {
			return nil, err
		}
		out.Delay = d
	}
	if sys.InputDelay != nil {
		id, err := convertSliceDelayToDiscrete(sys.InputDelay, dt, 0)
		if err != nil {
			return nil, err
		}
		out.InputDelay = id
	}
	if sys.OutputDelay != nil {
		od, err := convertSliceDelayToDiscrete(sys.OutputDelay, dt, 0)
		if err != nil {
			return nil, err
		}
		out.OutputDelay = od
	}
	propagateNames(out, sys)
	return out, nil
}

func discretizeImpulseAugmented(sys *System, dt float64) (*System, *mat.Dense, error) {
	n, m, _ := sys.Dims()
	N := len(sys.InternalDelay)

	C := denseCopy(sys.C)
	D := denseCopy(sys.D)

	if n == 0 {
		out := &System{A: newDense(0, 0), B: denseCopy(sys.B), C: C, D: D, Dt: dt}
		return out, mat.DenseCopyOf(sys.B2), nil
	}

	Adt := mat.NewDense(n, n, nil)
	Adt.Scale(dt, sys.A)
	var eA mat.Dense
	eA.Exp(Adt)
	Ad := mat.NewDense(n, n, nil)
	Ad.Copy(&eA)

	var Bd *mat.Dense
	if m > 0 {
		Bd = mat.NewDense(n, m, nil)
		Bd.Mul(Ad, sys.B)
		Bd.Scale(dt, Bd)
	} else {
		Bd = newDense(n, 0)
	}

	var Bd2 *mat.Dense
	if N > 0 {
		Bd2 = mat.NewDense(n, N, nil)
		Bd2.Mul(Ad, sys.B2)
		Bd2.Scale(dt, Bd2)
	} else {
		Bd2 = newDense(n, 0)
	}

	out := &System{A: Ad, B: Bd, C: C, D: D, Dt: dt}
	return out, Bd2, nil
}

func (sys *System) DiscretizeFOH(dt float64) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("DiscretizeFOH: system already discrete: %w", ErrWrongDomain)
	}
	if dt <= 0 {
		return nil, ErrInvalidSampleTime
	}

	if sys.HasInternalDelay() {
		return discretizeWithInternalDelay(sys, dt, C2DOptions{Method: "foh"})
	}

	n, m, p := sys.Dims()

	if n == 0 {
		return fohPureGain(sys, m, p, dt)
	}

	if m == 0 {
		Adt := mat.NewDense(n, n, nil)
		Adt.Scale(dt, sys.A)
		var eA mat.Dense
		eA.Exp(Adt)
		Ad := mat.NewDense(n, n, nil)
		Ad.Copy(&eA)
		out := &System{A: Ad, B: denseCopy(sys.B), C: denseCopy(sys.C), D: denseCopy(sys.D), Dt: dt}
		propagateNames(out, sys)
		return out, nil
	}

	nm := n + 2*m
	M := mat.NewDense(nm, nm, nil)
	mRaw := M.RawMatrix()
	aRaw := sys.A.RawMatrix()
	bRaw := sys.B.RawMatrix()

	for i := 0; i < n; i++ {
		row := mRaw.Data[i*mRaw.Stride:]
		for j := 0; j < n; j++ {
			row[j] = aRaw.Data[i*aRaw.Stride+j] * dt
		}
		for j := 0; j < m; j++ {
			row[n+j] = bRaw.Data[i*bRaw.Stride+j] * dt
		}
	}
	for i := 0; i < m; i++ {
		mRaw.Data[(n+i)*mRaw.Stride+n+m+i] = 1
	}

	var eM mat.Dense
	eM.Exp(M)
	emRaw := eM.RawMatrix()

	adData := make([]float64, n*n)
	g0Data := make([]float64, n*m)
	g1Data := make([]float64, n*m)
	for i := 0; i < n; i++ {
		copy(adData[i*n:i*n+n], emRaw.Data[i*emRaw.Stride:i*emRaw.Stride+n])
		copy(g0Data[i*m:i*m+m], emRaw.Data[i*emRaw.Stride+n:i*emRaw.Stride+n+m])
		copy(g1Data[i*m:i*m+m], emRaw.Data[i*emRaw.Stride+n+m:i*emRaw.Stride+n+2*m])
	}

	return buildFOHSystem(sys, n, m, p, dt, adData, g0Data, g1Data)
}

func fohPureGain(sys *System, m, p int, dt float64) (*System, error) {
	nNew := m
	aData := make([]float64, nNew*nNew)
	bData := make([]float64, nNew*m)
	for j := 0; j < m; j++ {
		bData[j*m+j] = 1
	}
	cData := make([]float64, p*nNew)
	if sys.D != nil {
		dRaw := sys.D.RawMatrix()
		for i := 0; i < p; i++ {
			copy(cData[i*nNew:i*nNew+m], dRaw.Data[i*dRaw.Stride:i*dRaw.Stride+m])
		}
	}
	dData := make([]float64, p*m)

	out := &System{
		A:  mat.NewDense(nNew, nNew, aData),
		B:  mat.NewDense(nNew, m, bData),
		C:  mat.NewDense(p, nNew, cData),
		D:  mat.NewDense(p, m, dData),
		Dt: dt,
	}
	propagateNames(out, sys)
	fohAugmentStateNames(out, sys, 0, m)
	return out, nil
}

func buildFOHSystem(sys *System, n, m, p int, dt float64, adData, g0Data, g1Data []float64) (*System, error) {
	b0Data := make([]float64, n*m)
	for i := range g0Data {
		b0Data[i] = g0Data[i] - g1Data[i]
	}

	nNew := n + m
	aFoh := make([]float64, nNew*nNew)
	for i := 0; i < n; i++ {
		copy(aFoh[i*nNew:i*nNew+n], adData[i*n:i*n+n])
		copy(aFoh[i*nNew+n:i*nNew+n+m], b0Data[i*m:i*m+m])
	}

	bFoh := make([]float64, nNew*m)
	for i := 0; i < n; i++ {
		copy(bFoh[i*m:i*m+m], g1Data[i*m:i*m+m])
	}
	for j := 0; j < m; j++ {
		bFoh[(n+j)*m+j] = 1
	}

	cFoh := make([]float64, p*nNew)
	if sys.C != nil {
		cRaw := sys.C.RawMatrix()
		for i := 0; i < p; i++ {
			copy(cFoh[i*nNew:i*nNew+n], cRaw.Data[i*cRaw.Stride:i*cRaw.Stride+n])
		}
	}
	if sys.D != nil {
		dRaw := sys.D.RawMatrix()
		for i := 0; i < p; i++ {
			copy(cFoh[i*nNew+n:i*nNew+n+m], dRaw.Data[i*dRaw.Stride:i*dRaw.Stride+m])
		}
	}

	dFoh := make([]float64, p*m)

	out := &System{
		A:  mat.NewDense(nNew, nNew, aFoh),
		B:  mat.NewDense(nNew, m, bFoh),
		C:  mat.NewDense(p, nNew, cFoh),
		D:  mat.NewDense(p, m, dFoh),
		Dt: dt,
	}

	if sys.Delay != nil {
		d, err := convertDelayToDiscrete(sys.Delay, dt)
		if err != nil {
			return nil, err
		}
		out.Delay = d
	}
	if sys.InputDelay != nil {
		id, err := convertSliceDelayToDiscrete(sys.InputDelay, dt, 0)
		if err != nil {
			return nil, err
		}
		out.InputDelay = id
	}
	if sys.OutputDelay != nil {
		od, err := convertSliceDelayToDiscrete(sys.OutputDelay, dt, 0)
		if err != nil {
			return nil, err
		}
		out.OutputDelay = od
	}
	propagateNames(out, sys)
	fohAugmentStateNames(out, sys, n, m)
	return out, nil
}

func fohAugmentStateNames(out, src *System, n, m int) {
	if src.StateName == nil && src.InputName == nil {
		return
	}
	names := make([]string, n+m)
	if src.StateName != nil {
		copy(names, src.StateName)
	}
	for j := 0; j < m; j++ {
		if src.InputName != nil && j < len(src.InputName) {
			names[n+j] = src.InputName[j] + "_prev"
		}
	}
	out.StateName = names
}

func discretizeFOHAugmented(sys *System, dt float64) (*System, *mat.Dense, error) {
	n, m, p := sys.Dims()
	N := len(sys.InternalDelay)
	mTotal := m + N

	C := denseCopy(sys.C)
	D := denseCopy(sys.D)

	if n == 0 {
		out, err := fohPureGain(sys, m, p, dt)
		if err != nil {
			return nil, nil, err
		}
		return out, mat.DenseCopyOf(sys.B2), nil
	}

	if mTotal == 0 {
		Adt := mat.NewDense(n, n, nil)
		Adt.Scale(dt, sys.A)
		var eA mat.Dense
		eA.Exp(Adt)
		Ad := mat.NewDense(n, n, nil)
		Ad.Copy(&eA)
		out := &System{A: Ad, B: denseCopy(sys.B), C: C, D: D, Dt: dt}
		return out, mat.DenseCopyOf(sys.B2), nil
	}

	sz := n + 2*mTotal
	M := mat.NewDense(sz, sz, nil)
	mRaw := M.RawMatrix()
	aRaw := sys.A.RawMatrix()

	for i := 0; i < n; i++ {
		row := mRaw.Data[i*mRaw.Stride:]
		for j := 0; j < n; j++ {
			row[j] = aRaw.Data[i*aRaw.Stride+j] * dt
		}
		if m > 0 {
			bRaw := sys.B.RawMatrix()
			for j := 0; j < m; j++ {
				row[n+j] = bRaw.Data[i*bRaw.Stride+j] * dt
			}
		}
		if N > 0 {
			b2Raw := sys.B2.RawMatrix()
			for j := 0; j < N; j++ {
				row[n+m+j] = b2Raw.Data[i*b2Raw.Stride+j] * dt
			}
		}
	}
	for i := 0; i < mTotal; i++ {
		mRaw.Data[(n+i)*mRaw.Stride+n+mTotal+i] = 1
	}

	var eM mat.Dense
	eM.Exp(M)
	emRaw := eM.RawMatrix()

	adData := make([]float64, n*n)
	g0BData := make([]float64, n*m)
	g1BData := make([]float64, n*m)
	g0B2Data := make([]float64, n*N)
	g1B2Data := make([]float64, n*N)
	for i := 0; i < n; i++ {
		copy(adData[i*n:i*n+n], emRaw.Data[i*emRaw.Stride:i*emRaw.Stride+n])
		if m > 0 {
			copy(g0BData[i*m:i*m+m], emRaw.Data[i*emRaw.Stride+n:i*emRaw.Stride+n+m])
			copy(g1BData[i*m:i*m+m], emRaw.Data[i*emRaw.Stride+n+mTotal:i*emRaw.Stride+n+mTotal+m])
		}
		if N > 0 {
			copy(g0B2Data[i*N:i*N+N], emRaw.Data[i*emRaw.Stride+n+m:i*emRaw.Stride+n+m+N])
			copy(g1B2Data[i*N:i*N+N], emRaw.Data[i*emRaw.Stride+n+mTotal+m:i*emRaw.Stride+n+mTotal+m+N])
		}
	}

	out, err := buildFOHSystem(sys, n, m, p, dt, adData, g0BData, g1BData)
	if err != nil {
		return nil, nil, err
	}

	var Bd2 *mat.Dense
	if N > 0 {
		b0B2 := make([]float64, n*N)
		for i := range g0B2Data {
			b0B2[i] = g0B2Data[i] - g1B2Data[i]
		}
		_ = b0B2
		Bd2 = mat.NewDense(n, N, g1B2Data)
	} else {
		Bd2 = newDense(n, 0)
	}

	return out, Bd2, nil
}

func (sys *System) DiscretizeMatched(dt float64) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("DiscretizeMatched: system already discrete: %w", ErrWrongDomain)
	}
	if dt <= 0 {
		return nil, ErrInvalidSampleTime
	}

	n, m, p := sys.Dims()
	if m != 1 || p != 1 {
		return nil, fmt.Errorf("DiscretizeMatched: %w", ErrNotSISO)
	}

	if sys.HasInternalDelay() {
		return nil, fmt.Errorf("DiscretizeMatched: internal delays not supported: %w", ErrFeedbackDelay)
	}

	if n == 0 {
		out := &System{
			A:  newDense(0, 0),
			B:  denseCopy(sys.B),
			C:  denseCopy(sys.C),
			D:  denseCopy(sys.D),
			Dt: dt,
		}
		propagateNames(out, sys)
		return out, nil
	}

	contPoles, err := sys.Poles()
	if err != nil {
		return nil, fmt.Errorf("DiscretizeMatched: %w", err)
	}
	contZeros, err := sys.Zeros()
	if err != nil {
		return nil, fmt.Errorf("DiscretizeMatched: %w", err)
	}

	discPoles := make([]complex128, len(contPoles))
	for i, p := range contPoles {
		discPoles[i] = cmplx.Exp(p * complex(dt, 0))
	}
	discZeros := make([]complex128, len(contZeros), len(contZeros)+len(contPoles))
	for i, z := range contZeros {
		discZeros[i] = cmplx.Exp(z * complex(dt, 0))
	}
	excess := len(contPoles) - len(contZeros)
	for i := 0; i < excess; i++ {
		discZeros = append(discZeros, complex(-1, 0))
	}

	gain, err := matchedGain(sys, discZeros, discPoles, dt)
	if err != nil {
		return nil, err
	}

	zpk, err := NewZPK(discZeros, discPoles, gain, dt)
	if err != nil {
		return nil, fmt.Errorf("DiscretizeMatched: %w", err)
	}
	ssResult, err := zpk.StateSpace(nil)
	if err != nil {
		return nil, fmt.Errorf("DiscretizeMatched: %w", err)
	}
	result := ssResult.Sys

	if sys.Delay != nil {
		d, err := convertDelayToDiscrete(sys.Delay, dt)
		if err != nil {
			return nil, err
		}
		result.Delay = d
	}
	if sys.InputDelay != nil {
		id, err := convertSliceDelayToDiscrete(sys.InputDelay, dt, 0)
		if err != nil {
			return nil, err
		}
		result.InputDelay = id
	}
	if sys.OutputDelay != nil {
		od, err := convertSliceDelayToDiscrete(sys.OutputDelay, dt, 0)
		if err != nil {
			return nil, err
		}
		result.OutputDelay = od
	}
	propagateNames(result, sys)
	return result, nil
}

func matchedGain(sys *System, discZeros, discPoles []complex128, dt float64) (float64, error) {
	tfr, err := sys.TransferFunction(nil)
	if err != nil {
		return 0, fmt.Errorf("DiscretizeMatched: %w", err)
	}
	num := Poly(tfr.TF.Num[0][0])
	den := Poly(tfr.TF.Den[0])

	denAt0 := den.Eval(0)
	if cmplx.Abs(denAt0) > 1e-10 {
		contDC := num.Eval(0) / denAt0
		discDC := zpkEvalChannel(complex(1, 0), discZeros, discPoles, 1.0)
		if cmplx.Abs(discDC) < 1e-14 {
			return matchedGainFallback(num, den, discZeros, discPoles, dt)
		}
		return real(contDC / discDC), nil
	}

	return matchedGainFallback(num, den, discZeros, discPoles, dt)
}

func matchedGainFallback(num, den Poly, discZeros, discPoles []complex128, dt float64) (float64, error) {
	sMatch := complex(0, math.Pi/(2*dt))
	zMatch := complex(0, 1)

	contVal := num.Eval(sMatch) / den.Eval(sMatch)
	discVal := zpkEvalChannel(zMatch, discZeros, discPoles, 1.0)

	if cmplx.Abs(discVal) < 1e-14 {
		sMatch = complex(0, math.Pi/(4*dt))
		zMatch = cmplx.Exp(complex(0, math.Pi/4))
		contVal = num.Eval(sMatch) / den.Eval(sMatch)
		discVal = zpkEvalChannel(zMatch, discZeros, discPoles, 1.0)
	}
	if cmplx.Abs(discVal) < 1e-14 {
		return 0, fmt.Errorf("DiscretizeMatched: cannot determine gain: %w", ErrSingularTransform)
	}
	return cmplx.Abs(contVal) / cmplx.Abs(discVal), nil
}

func (sys *System) D2D(newDt float64, opts C2DOptions) (*System, error) {
	if sys.IsContinuous() {
		return nil, fmt.Errorf("D2D: system is continuous: %w", ErrWrongDomain)
	}
	if newDt <= 0 {
		return nil, ErrInvalidSampleTime
	}
	if math.Abs(newDt-sys.Dt) < 1e-14*math.Max(newDt, sys.Dt) {
		return sys.Copy(), nil
	}

	contSys, err := sys.Undiscretize()
	if err != nil {
		return nil, fmt.Errorf("D2D: %w", err)
	}

	result, err := contSys.DiscretizeWithOpts(newDt, opts)
	if err != nil {
		return nil, fmt.Errorf("D2D: %w", err)
	}
	propagateNames(result, sys)
	return result, nil
}
