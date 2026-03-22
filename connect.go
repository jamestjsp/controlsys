package controlsys

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func domainMatch(sys1, sys2 *System) error {
	if sys1.Dt != sys2.Dt {
		return ErrDomainMismatch
	}
	return nil
}

func setBlock(dst *mat.Dense, r0, c0 int, src *mat.Dense) {
	if src == nil {
		return
	}
	sr, sc := src.Dims()
	if sr == 0 || sc == 0 {
		return
	}
	dRaw := dst.RawMatrix()
	sRaw := src.RawMatrix()
	for i := 0; i < sr; i++ {
		copy(dRaw.Data[(r0+i)*dRaw.Stride+c0:], sRaw.Data[i*sRaw.Stride:i*sRaw.Stride+sc])
	}
}

func Series(sys1, sys2 *System) (*System, error) {
	if err := domainMatch(sys1, sys2); err != nil {
		return nil, err
	}
	_, _, p1 := sys1.Dims()
	_, m2, _ := sys2.Dims()
	if p1 != m2 {
		return nil, fmt.Errorf("series: sys1 outputs %d != sys2 inputs %d: %w", p1, m2, ErrDimensionMismatch)
	}

	if sys1.HasInternalDelay() || sys2.HasInternalDelay() {
		return seriesLFT(sys1, sys2)
	}

	needsLFT := false
	if sys1.HasDelay() || sys2.HasDelay() {
		ioCheck := seriesDelay(sys1, sys2)
		if ioCheck == nil && (sys1.Delay != nil || sys2.Delay != nil) {
			needsLFT = true
		}
		if !needsLFT && (sys1.OutputDelay != nil || sys2.InputDelay != nil) {
			for k := 0; k < p1; k++ {
				var od, id float64
				if sys1.OutputDelay != nil {
					od = sys1.OutputDelay[k]
				}
				if sys2.InputDelay != nil {
					id = sys2.InputDelay[k]
				}
				if k > 0 {
					var od0, id0 float64
					if sys1.OutputDelay != nil {
						od0 = sys1.OutputDelay[0]
					}
					if sys2.InputDelay != nil {
						id0 = sys2.InputDelay[0]
					}
					if math.Abs((od+id)-(od0+id0)) > 1e-12 {
						needsLFT = true
						break
					}
				}
			}
		}
	}

	if needsLFT {
		return seriesLFT(sys1, sys2)
	}

	return seriesSimple(sys1, sys2)
}

func seriesSimple(sys1, sys2 *System) (*System, error) {
	n1, m1, _ := sys1.Dims()
	n2, _, p2 := sys2.Dims()
	n := n1 + n2
	m := m1
	p := p2

	ioDelay := seriesDelay(sys1, sys2)
	inDel, outDel, ioDelay := seriesInputOutputDelay(sys1, sys2, p, m, ioDelay)

	buildSeries := func(A, B, C, D *mat.Dense) (*System, error) {
		sys, err := buildSystem(A, B, C, D, sys1.Dt, ioDelay)
		if err != nil {
			return nil, err
		}
		sys.InputDelay = inDel
		sys.OutputDelay = outDel
		sys.InputName = copyStringSlice(sys1.InputName)
		sys.OutputName = copyStringSlice(sys2.OutputName)
		sys.StateName = concatStringSlices([][]string{sys1.StateName, sys2.StateName}, []int{n1, n2})
		return sys, nil
	}

	if n1 == 0 && n2 == 0 {
		D := mat.NewDense(p, m, nil)
		D.Mul(sys2.D, sys1.D)
		return buildSeries(nil, nil, nil, D)
	}

	if n1 == 0 {
		A := denseCopy(sys2.A)
		B := mat.NewDense(n2, m, nil)
		B.Mul(sys2.B, sys1.D)
		C := denseCopy(sys2.C)
		D := mat.NewDense(p, m, nil)
		D.Mul(sys2.D, sys1.D)
		return buildSeries(A, B, C, D)
	}

	if n2 == 0 {
		A := denseCopy(sys1.A)
		B := denseCopy(sys1.B)
		C := mat.NewDense(p, n1, nil)
		C.Mul(sys2.D, sys1.C)
		D := mat.NewDense(p, m, nil)
		D.Mul(sys2.D, sys1.D)
		return buildSeries(A, B, C, D)
	}

	A := mat.NewDense(n, n, nil)
	setBlock(A, 0, 0, sys1.A)
	setBlock(A, n1, 0, mulDense(sys2.B, sys1.C))
	setBlock(A, n1, n1, sys2.A)

	B := mat.NewDense(n, m, nil)
	setBlock(B, 0, 0, sys1.B)
	setBlock(B, n1, 0, mulDense(sys2.B, sys1.D))

	C := mat.NewDense(p, n, nil)
	setBlock(C, 0, 0, mulDense(sys2.D, sys1.C))
	setBlock(C, 0, n1, sys2.C)

	D := mat.NewDense(p, m, nil)
	D.Mul(sys2.D, sys1.D)

	return buildSeries(A, B, C, D)
}

func seriesDelay(sys1, sys2 *System) *mat.Dense {
	if sys1.Delay == nil && sys2.Delay == nil {
		return nil
	}
	_, m1, p1 := sys1.Dims()
	_, _, p2 := sys2.Dims()

	d1 := ioDelayOrZero(sys1.Delay, p1, m1)
	d2 := ioDelayOrZero(sys2.Delay, p2, p1)

	result := mat.NewDense(p2, m1, nil)
	for i := 0; i < p2; i++ {
		for j := 0; j < m1; j++ {
			ref := d1.At(0, j) + d2.At(i, 0)
			for k := 1; k < p1; k++ {
				if math.Abs(d1.At(k, j)+d2.At(i, k)-ref) > 1e-12 {
					return nil
				}
			}
			result.Set(i, j, ref)
		}
	}
	return result
}

func seriesInputOutputDelay(sys1, sys2 *System, p, m int, ioDelay *mat.Dense) (inputDelay, outputDelay []float64, resultIO *mat.Dense) {
	_, _, p1 := sys1.Dims()

	inputDelay = copySliceOrNil(sys1.InputDelay)
	outputDelay = copySliceOrNil(sys2.OutputDelay)

	intermediate := make([]float64, p1)
	hasIntermediate := false
	for k := 0; k < p1; k++ {
		var od, id float64
		if sys1.OutputDelay != nil {
			od = sys1.OutputDelay[k]
		}
		if sys2.InputDelay != nil {
			id = sys2.InputDelay[k]
		}
		intermediate[k] = od + id
		if intermediate[k] != 0 {
			hasIntermediate = true
		}
	}

	if !hasIntermediate {
		return inputDelay, outputDelay, ioDelay
	}

	minIntermediate := intermediate[0]
	for k := 1; k < p1; k++ {
		minIntermediate = math.Min(minIntermediate, intermediate[k])
	}

	if ioDelay == nil {
		ioDelay = mat.NewDense(p, m, nil)
	} else {
		ioDelay = mat.DenseCopyOf(ioDelay)
	}

	raw := ioDelay.RawMatrix()
	for i := 0; i < p; i++ {
		for j := 0; j < m; j++ {
			raw.Data[i*raw.Stride+j] += minIntermediate
		}
	}

	return inputDelay, outputDelay, ioDelay
}

func copySliceOrNil(s []float64) []float64 {
	if s == nil {
		return nil
	}
	out := make([]float64, len(s))
	copy(out, s)
	return out
}

func ioDelayOrZero(delay *mat.Dense, p, m int) *mat.Dense {
	if delay != nil {
		return delay
	}
	return mat.NewDense(p, m, nil)
}

func Parallel(sys1, sys2 *System) (*System, error) {
	if err := domainMatch(sys1, sys2); err != nil {
		return nil, err
	}
	_, m1, p1 := sys1.Dims()
	_, m2, p2 := sys2.Dims()
	if m1 != m2 || p1 != p2 {
		return nil, fmt.Errorf("parallel: dims (%d,%d) vs (%d,%d): %w", p1, m1, p2, m2, ErrDimensionMismatch)
	}

	if sys1.HasInternalDelay() || sys2.HasInternalDelay() {
		return parallelLFT(sys1, sys2)
	}

	if parallelNeedsLFT(sys1, sys2) {
		return parallelLFT(sys1, sys2)
	}

	return parallelSimple(sys1, sys2)
}

func parallelNeedsLFT(sys1, sys2 *System) bool {
	if !sys1.HasDelay() && !sys2.HasDelay() {
		return false
	}
	_, m, p := sys1.Dims()

	td1 := totalDelayMatrix(sys1, p, m)
	td2 := totalDelayMatrix(sys2, p, m)
	if td1 == nil && td2 == nil {
		return false
	}
	if td1 == nil {
		td1 = mat.NewDense(p, m, nil)
	}
	if td2 == nil {
		td2 = mat.NewDense(p, m, nil)
	}
	return !mat.Equal(td1, td2)
}

func totalDelayMatrix(sys *System, p, m int) *mat.Dense {
	if sys.Delay == nil && sys.InputDelay == nil && sys.OutputDelay == nil {
		return nil
	}
	data := make([]float64, p*m)
	if sys.Delay != nil {
		raw := sys.Delay.RawMatrix()
		for i := 0; i < p; i++ {
			copy(data[i*m:i*m+m], raw.Data[i*raw.Stride:i*raw.Stride+m])
		}
	}
	if sys.InputDelay != nil {
		for j := 0; j < m; j++ {
			for i := 0; i < p; i++ {
				data[i*m+j] += sys.InputDelay[j]
			}
		}
	}
	if sys.OutputDelay != nil {
		for i := 0; i < p; i++ {
			for j := 0; j < m; j++ {
				data[i*m+j] += sys.OutputDelay[i]
			}
		}
	}
	return mat.NewDense(p, m, data)
}

func parallelSimple(sys1, sys2 *System) (*System, error) {
	n1, m1, p1 := sys1.Dims()
	n2, _, _ := sys2.Dims()
	n := n1 + n2
	m := m1
	p := p1

	ioDelay := parallelDelay(sys1, sys2)
	inDel, outDel, ioDelay := parallelInputOutputDelay(sys1, sys2, m, p, ioDelay)

	buildParallel := func(A, B, C, D *mat.Dense) (*System, error) {
		sys, err := buildSystem(A, B, C, D, sys1.Dt, ioDelay)
		if err != nil {
			return nil, err
		}
		sys.InputDelay = inDel
		sys.OutputDelay = outDel
		sys.InputName = copyStringSlice(sys1.InputName)
		sys.OutputName = copyStringSlice(sys1.OutputName)
		sys.StateName = concatStringSlices([][]string{sys1.StateName, sys2.StateName}, []int{n1, n2})
		return sys, nil
	}

	if n1 == 0 && n2 == 0 {
		D := mat.NewDense(p, m, nil)
		D.Add(sys1.D, sys2.D)
		return buildParallel(nil, nil, nil, D)
	}

	if n1 == 0 {
		A := denseCopy(sys2.A)
		B := denseCopy(sys2.B)
		C := denseCopy(sys2.C)
		D := mat.NewDense(p, m, nil)
		D.Add(sys1.D, sys2.D)
		return buildParallel(A, B, C, D)
	}

	if n2 == 0 {
		A := denseCopy(sys1.A)
		B := denseCopy(sys1.B)
		C := denseCopy(sys1.C)
		D := mat.NewDense(p, m, nil)
		D.Add(sys1.D, sys2.D)
		return buildParallel(A, B, C, D)
	}

	A := mat.NewDense(n, n, nil)
	setBlock(A, 0, 0, sys1.A)
	setBlock(A, n1, n1, sys2.A)

	B := mat.NewDense(n, m, nil)
	setBlock(B, 0, 0, sys1.B)
	setBlock(B, n1, 0, sys2.B)

	C := mat.NewDense(p, n, nil)
	setBlock(C, 0, 0, sys1.C)
	setBlock(C, 0, n1, sys2.C)

	D := mat.NewDense(p, m, nil)
	D.Add(sys1.D, sys2.D)

	return buildParallel(A, B, C, D)
}

func parallelDelay(sys1, sys2 *System) *mat.Dense {
	if sys1.Delay == nil && sys2.Delay == nil {
		return nil
	}
	_, m, p := sys1.Dims()
	d1 := ioDelayOrZero(sys1.Delay, p, m)
	d2 := ioDelayOrZero(sys2.Delay, p, m)
	if mat.Equal(d1, d2) {
		return mat.DenseCopyOf(d1)
	}
	return nil
}

func parallelInputOutputDelay(sys1, sys2 *System, m, p int, ioDelay *mat.Dense) (inputDelay, outputDelay []float64, resultIO *mat.Dense) {
	resultIO = ioDelay
	inputDelay = mergeParallelDelay(sys1.InputDelay, sys2.InputDelay, m)
	outputDelay = mergeParallelDelay(sys1.OutputDelay, sys2.OutputDelay, p)

	in1 := sliceOrZeros(sys1.InputDelay, m)
	in2 := sliceOrZeros(sys2.InputDelay, m)
	out1 := sliceOrZeros(sys1.OutputDelay, p)
	out2 := sliceOrZeros(sys2.OutputDelay, p)

	commonIn := sliceOrZeros(inputDelay, m)
	commonOut := sliceOrZeros(outputDelay, p)

	hasResidual := false
	residual1 := mat.NewDense(p, m, nil)
	residual2 := mat.NewDense(p, m, nil)
	r1Raw := residual1.RawMatrix()
	r2Raw := residual2.RawMatrix()
	for i := 0; i < p; i++ {
		for j := 0; j < m; j++ {
			v1 := (in1[j] - commonIn[j]) + (out1[i] - commonOut[i])
			v2 := (in2[j] - commonIn[j]) + (out2[i] - commonOut[i])
			r1Raw.Data[i*r1Raw.Stride+j] = v1
			r2Raw.Data[i*r2Raw.Stride+j] = v2
			if v1 != 0 || v2 != 0 {
				hasResidual = true
			}
		}
	}

	if !hasResidual {
		return
	}

	if !mat.Equal(residual1, residual2) {
		if resultIO == nil {
			resultIO = mat.NewDense(p, m, nil)
		} else {
			resultIO = mat.DenseCopyOf(resultIO)
		}
	}

	return
}

func mergeParallelDelay(d1, d2 []float64, n int) []float64 {
	if d1 == nil && d2 == nil {
		return nil
	}
	s1 := sliceOrZeros(d1, n)
	s2 := sliceOrZeros(d2, n)
	out := make([]float64, n)
	allZero := true
	for i := 0; i < n; i++ {
		out[i] = math.Min(s1[i], s2[i])
		if out[i] != 0 {
			allZero = false
		}
	}
	if allZero && d1 == nil && d2 == nil {
		return nil
	}
	return out
}

func sliceOrZeros(s []float64, n int) []float64 {
	if s != nil {
		return s
	}
	return make([]float64, n)
}

func Feedback(plant, controller *System, sign float64) (*System, error) {
	if plant == nil {
		return nil, fmt.Errorf("feedback: plant cannot be nil")
	}
	if controller == nil {
		_, p, m := plant.Dims()
		if p != m {
			return nil, fmt.Errorf("feedback: plant must be square (m=p) when controller is nil, got %dx%d", m, p)
		}
		eyeData := make([]float64, m*m)
		for i := 0; i < m; i++ {
			eyeData[i*(m+1)] = 1
		}
		var err error
		controller, err = NewGain(mat.NewDense(m, m, eyeData), plant.Dt)
		if err != nil {
			return nil, err
		}
	}
	if err := domainMatch(plant, controller); err != nil {
		return nil, err
	}
	n1, m1, p1 := plant.Dims()
	n2, m2, p2 := controller.Dims()
	if p1 != m2 {
		return nil, fmt.Errorf("feedback: plant outputs %d != controller inputs %d: %w", p1, m2, ErrDimensionMismatch)
	}
	if m1 != p2 {
		return nil, fmt.Errorf("feedback: plant inputs %d != controller outputs %d: %w", m1, p2, ErrDimensionMismatch)
	}
	if plant.HasDelay() || controller.HasDelay() {
		return feedbackWithLFT(plant, controller, sign)
	}

	// User convention: sign=-1 for negative feedback, +1 for positive.
	// AB05ND formula uses opposite convention internally.
	sign = -sign

	n := n1 + n2
	m := m1
	p := p1

	M := mat.NewDense(p1, p1, nil)
	M.Mul(plant.D, controller.D)
	M.Scale(sign, M)
	for i := 0; i < p1; i++ {
		M.Set(i, i, M.At(i, i)+1)
	}

	var lu mat.LU
	lu.Factorize(M)
	if lu.Det() == 0 {
		return nil, fmt.Errorf("feedback: (I + sign*D1*D2) singular: %w", ErrSingularTransform)
	}

	eyeData := make([]float64, p1*p1)
	for i := 0; i < p1; i++ {
		eyeData[i*(p1+1)] = 1
	}
	eye := mat.NewDense(p1, p1, eyeData)
	E21 := mat.NewDense(p1, p1, nil)
	if err := lu.SolveTo(E21, false, eye); err != nil {
		return nil, fmt.Errorf("feedback: LU solve failed: %w", ErrSingularTransform)
	}

	e12Data := make([]float64, m1*m1)
	for i := 0; i < m1; i++ {
		e12Data[i*(m1+1)] = 1
	}
	E12 := mat.NewDense(m1, m1, e12Data)
	tmp := mat.NewDense(m1, p1, nil)
	tmp.Mul(controller.D, E21)
	tmp2 := mat.NewDense(m1, m1, nil)
	tmp2.Mul(tmp, plant.D)
	tmp2.Scale(sign, tmp2)
	E12.Sub(E12, tmp2)

	if n1 == 0 && n2 == 0 {
		D := mat.NewDense(p, m, nil)
		D.Mul(E21, plant.D)
		sys, err := buildSystem(nil, nil, nil, D, plant.Dt, nil)
		if err != nil {
			return nil, err
		}
		sys.InputName = copyStringSlice(plant.InputName)
		sys.OutputName = copyStringSlice(plant.OutputName)
		sys.StateName = concatStringSlices([][]string{plant.StateName, controller.StateName}, []int{n1, n2})
		return sys, nil
	}

	A := mat.NewDense(n, n, nil)
	Bres := mat.NewDense(n, m, nil)
	C := mat.NewDense(p, n, nil)
	D := mat.NewDense(p, m, nil)

	if n1 > 0 {
		B1E12 := mat.NewDense(n1, m1, nil)
		B1E12.Mul(plant.B, E12)

		sB1E12D2 := mat.NewDense(n1, p1, nil)
		sB1E12D2.Mul(B1E12, controller.D)
		sB1E12D2.Scale(sign, sB1E12D2)
		block := mat.NewDense(n1, n1, nil)
		block.Mul(sB1E12D2, plant.C)
		setBlock(A, 0, 0, plant.A)
		subBlock(A, 0, 0, block)

		if n2 > 0 {
			block2 := mat.NewDense(n1, n2, nil)
			block2.Mul(B1E12, controller.C)
			block2.Scale(-sign, block2)
			setBlock(A, 0, n1, block2)
		}

		setBlock(Bres, 0, 0, B1E12)
	}

	if n2 > 0 {
		B2E21 := mat.NewDense(n2, p1, nil)
		B2E21.Mul(controller.B, E21)

		if n1 > 0 {
			block := mat.NewDense(n2, n1, nil)
			block.Mul(B2E21, plant.C)
			setBlock(A, n1, 0, block)
		}

		setBlock(A, n1, n1, controller.A)
		B2E21D1 := mat.NewDense(n2, m1, nil)
		B2E21D1.Mul(B2E21, plant.D)

		sB2E21D1 := mat.NewDense(n2, m1, nil)
		sB2E21D1.Scale(sign, B2E21D1)
		block3 := mat.NewDense(n2, n2, nil)
		block3.Mul(sB2E21D1, controller.C)
		subBlock(A, n1, n1, block3)

		setBlock(Bres, n1, 0, B2E21D1)
	}

	if n1 > 0 {
		E21C1 := mat.NewDense(p1, n1, nil)
		E21C1.Mul(E21, plant.C)
		setBlock(C, 0, 0, E21C1)
	}
	if n2 > 0 {
		E21D1 := mat.NewDense(p1, m1, nil)
		E21D1.Mul(E21, plant.D)
		E21D1C2 := mat.NewDense(p1, n2, nil)
		E21D1C2.Mul(E21D1, controller.C)
		E21D1C2.Scale(-sign, E21D1C2)
		setBlock(C, 0, n1, E21D1C2)
	}

	D.Mul(E21, plant.D)

	sys, err := buildSystem(A, Bres, C, D, plant.Dt, nil)
	if err != nil {
		return nil, err
	}
	sys.InputName = copyStringSlice(plant.InputName)
	sys.OutputName = copyStringSlice(plant.OutputName)
	sys.StateName = concatStringSlices([][]string{plant.StateName, controller.StateName}, []int{n1, n2})
	return sys, nil
}

func subBlock(dst *mat.Dense, r0, c0 int, src *mat.Dense) {
	sr, sc := src.Dims()
	if sr == 0 || sc == 0 {
		return
	}
	dRaw := dst.RawMatrix()
	sRaw := src.RawMatrix()
	for i := 0; i < sr; i++ {
		dRow := dRaw.Data[(r0+i)*dRaw.Stride+c0:]
		sRow := sRaw.Data[i*sRaw.Stride:]
		for j := 0; j < sc; j++ {
			dRow[j] -= sRow[j]
		}
	}
}

func Append(sys1, sys2 *System) (*System, error) {
	if err := domainMatch(sys1, sys2); err != nil {
		return nil, err
	}
	n1, m1, p1 := sys1.Dims()
	n2, m2, p2 := sys2.Dims()

	n := n1 + n2
	m := m1 + m2
	p := p1 + p2

	if sys1.HasInternalDelay() || sys2.HasInternalDelay() {
		s1, err := sys1.PullDelaysToLFT()
		if err != nil {
			return nil, err
		}
		s2, err := sys2.PullDelaysToLFT()
		if err != nil {
			return nil, err
		}
		n1, m1, p1 = s1.Dims()
		n2, m2, p2 = s2.Dims()
		n, m, p = n1+n2, m1+m2, p1+p2
		res := &System{
			A:  mat.NewDense(max(n, 1), max(n, 1), nil),
			B:  mat.NewDense(max(n, 1), max(m, 1), nil),
			C:  mat.NewDense(max(p, 1), max(n, 1), nil),
			D:  mat.NewDense(max(p, 1), max(m, 1), nil),
			Dt: s1.Dt,
		}
		if n1 > 0 {
			setBlock(res.A, 0, 0, s1.A)
			setBlock(res.B, 0, 0, s1.B)
			setBlock(res.C, 0, 0, s1.C)
		}
		if n2 > 0 {
			setBlock(res.A, n1, n1, s2.A)
			setBlock(res.B, n1, m1, s2.B)
			setBlock(res.C, p1, n1, s2.C)
		}
		setBlock(res.D, 0, 0, s1.D)
		setBlock(res.D, p1, m1, s2.D)
		appendInternalDelay(res, s1, s2, n1, n2, m1, m2, p1, p2)
		res.InputName = concatStringSlices([][]string{sys1.InputName, sys2.InputName}, []int{m1, m2})
		res.OutputName = concatStringSlices([][]string{sys1.OutputName, sys2.OutputName}, []int{p1, p2})
		res.StateName = concatStringSlices([][]string{sys1.StateName, sys2.StateName}, []int{n1, n2})
		return res, nil
	}

	A := mat.NewDense(max(n, 1), max(n, 1), nil)
	B := mat.NewDense(max(n, 1), max(m, 1), nil)
	C := mat.NewDense(max(p, 1), max(n, 1), nil)
	D := mat.NewDense(max(p, 1), max(m, 1), nil)

	if n1 > 0 {
		setBlock(A, 0, 0, sys1.A)
		setBlock(B, 0, 0, sys1.B)
		setBlock(C, 0, 0, sys1.C)
	}
	if n2 > 0 {
		setBlock(A, n1, n1, sys2.A)
		setBlock(B, n1, m1, sys2.B)
		setBlock(C, p1, n1, sys2.C)
	}
	setBlock(D, 0, 0, sys1.D)
	setBlock(D, p1, m1, sys2.D)

	delay := appendDelay(sys1, sys2, p1, m1, p2, m2)
	inDel := concatDelay(sys1.InputDelay, m1, sys2.InputDelay, m2)
	outDel := concatDelay(sys1.OutputDelay, p1, sys2.OutputDelay, p2)

	if n == 0 {
		sys, _ := NewGain(D, sys1.Dt)
		sys.Delay = delay
		sys.InputDelay = inDel
		sys.OutputDelay = outDel
		appendInternalDelay(sys, sys1, sys2, n1, n2, m1, m2, p1, p2)
		sys.InputName = concatStringSlices([][]string{sys1.InputName, sys2.InputName}, []int{m1, m2})
		sys.OutputName = concatStringSlices([][]string{sys1.OutputName, sys2.OutputName}, []int{p1, p2})
		sys.StateName = concatStringSlices([][]string{sys1.StateName, sys2.StateName}, []int{n1, n2})
		return sys, nil
	}

	A = resizeDense(A, n, n)
	B = resizeDense(B, n, m)
	C = resizeDense(C, p, n)
	D = resizeDense(D, p, m)

	sys, err := buildSystem(A, B, C, D, sys1.Dt, delay)
	if err != nil {
		return nil, err
	}
	sys.InputDelay = inDel
	sys.OutputDelay = outDel
	appendInternalDelay(sys, sys1, sys2, n1, n2, m1, m2, p1, p2)
	sys.InputName = concatStringSlices([][]string{sys1.InputName, sys2.InputName}, []int{m1, m2})
	sys.OutputName = concatStringSlices([][]string{sys1.OutputName, sys2.OutputName}, []int{p1, p2})
	sys.StateName = concatStringSlices([][]string{sys1.StateName, sys2.StateName}, []int{n1, n2})
	return sys, nil
}

func resizeDense(src *mat.Dense, r, c int) *mat.Dense {
	sr, sc := src.Dims()
	if sr == r && sc == c {
		return src
	}
	if r == 0 || c == 0 {
		return &mat.Dense{}
	}
	dst := mat.NewDense(r, c, nil)
	minR := min(sr, r)
	minC := min(sc, c)
	sRaw := src.RawMatrix()
	dRaw := dst.RawMatrix()
	for i := 0; i < minR; i++ {
		copy(dRaw.Data[i*dRaw.Stride:i*dRaw.Stride+minC], sRaw.Data[i*sRaw.Stride:i*sRaw.Stride+minC])
	}
	return dst
}

func appendDelay(sys1, sys2 *System, p1, m1, p2, m2 int) *mat.Dense {
	if sys1.Delay == nil && sys2.Delay == nil {
		return nil
	}
	p := p1 + p2
	m := m1 + m2
	d := mat.NewDense(p, m, nil)
	if sys1.Delay != nil {
		setBlock(d, 0, 0, sys1.Delay)
	}
	if sys2.Delay != nil {
		setBlock(d, p1, m1, sys2.Delay)
	}
	return d
}

func concatDelay(d1 []float64, n1 int, d2 []float64, n2 int) []float64 {
	if d1 == nil && d2 == nil {
		return nil
	}
	out := make([]float64, n1+n2)
	if d1 != nil {
		copy(out, d1)
	}
	if d2 != nil {
		copy(out[n1:], d2)
	}
	return out
}

func mulDense(a, b *mat.Dense) *mat.Dense {
	ar, _ := a.Dims()
	_, bc := b.Dims()
	r := mat.NewDense(ar, bc, nil)
	r.Mul(a, b)
	return r
}

func seriesLFT(sys1, sys2 *System) (*System, error) {
	savedInput1 := copySliceOrNil(sys1.InputDelay)
	savedOutput2 := copySliceOrNil(sys2.OutputDelay)

	s1Stripped := sys1.Copy()
	s1Stripped.InputDelay = nil
	s2Stripped := sys2.Copy()
	s2Stripped.OutputDelay = nil

	s1, err := s1Stripped.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}
	s2, err := s2Stripped.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}

	n1, m1, _ := s1.Dims()
	n2, _, p2 := s2.Dims()
	N1 := s1.internalDelayCount()
	N2 := s2.internalDelayCount()
	N := N1 + N2
	n := n1 + n2
	m := m1
	p := p2

	tau := make([]float64, 0, N)
	if s1.LFT != nil {
		tau = append(tau, s1.LFT.Tau...)
	}
	if s2.LFT != nil {
		tau = append(tau, s2.LFT.Tau...)
	}

	A := mat.NewDense(max(n, 1), max(n, 1), nil)
	B := mat.NewDense(max(n, 1), max(m, 1), nil)
	C := mat.NewDense(max(p, 1), max(n, 1), nil)
	D := mat.NewDense(max(p, 1), max(m, 1), nil)

	if n1 > 0 {
		setBlock(A, 0, 0, s1.A)
		setBlock(B, 0, 0, s1.B)
	}
	if n2 > 0 {
		setBlock(A, n1, n1, s2.A)
	}
	if n1 > 0 && n2 > 0 {
		setBlock(A, n1, 0, mulDense(s2.B, s1.C))
	}
	if n2 > 0 {
		setBlock(B, n1, 0, mulDense(s2.B, s1.D))
	}
	if n1 > 0 {
		setBlock(C, 0, 0, mulDense(s2.D, s1.C))
	}
	if n2 > 0 {
		setBlock(C, 0, n1, s2.C)
	}
	D.Mul(s2.D, s1.D)

	A = resizeDense(A, n, n)
	B = resizeDense(B, n, m)
	C = resizeDense(C, p, n)
	D = resizeDense(D, p, m)

	if N == 0 {
		sys, err := newNoCopy(A, B, C, D, s1.Dt)
		if err != nil {
			return nil, err
		}
		sys.InputDelay = savedInput1
		sys.OutputDelay = savedOutput2
		sys.InputName = copyStringSlice(sys1.InputName)
		sys.OutputName = copyStringSlice(sys2.OutputName)
		sys.StateName = concatStringSlices([][]string{sys1.StateName, sys2.StateName}, []int{n1, n2})
		return sys, nil
	}

	b2 := mat.NewDense(max(n, 1), N, nil)
	c2 := mat.NewDense(N, max(n, 1), nil)
	d12 := mat.NewDense(max(p, 1), N, nil)
	d21 := mat.NewDense(N, max(m, 1), nil)
	d22 := mat.NewDense(N, N, nil)

	if N1 > 0 {
		if n1 > 0 {
			setBlock(b2, 0, 0, s1.LFT.B2)
		}
		if n2 > 0 {
			setBlock(b2, n1, 0, mulDense(s2.B, s1.LFT.D12))
		}
		if n1 > 0 {
			setBlock(c2, 0, 0, s1.LFT.C2)
		}
		setBlock(d12, 0, 0, mulDense(s2.D, s1.LFT.D12))
		setBlock(d21, 0, 0, s1.LFT.D21)
		if N2 > 0 {
			setBlock(d22, N1, 0, mulDense(s2.LFT.D21, s1.LFT.D12))
		}
	}

	if N2 > 0 {
		if n2 > 0 {
			setBlock(b2, n1, N1, s2.LFT.B2)
		}
		if n1 > 0 {
			setBlock(c2, N1, n1, s2.LFT.C2)
		}
		if n2 > 0 {
			c2sub := mulDense(s2.LFT.D21, s1.C)
			r, c := c2sub.Dims()
			for i := 0; i < r; i++ {
				for j := 0; j < c; j++ {
					c2.Set(N1+i, j, c2.At(N1+i, j)+c2sub.At(i, j))
				}
			}
		}
		setBlock(d12, 0, N1, s2.LFT.D12)
		d21sub := mulDense(s2.LFT.D21, s1.D)
		setBlock(d21, N1, 0, d21sub)
		setBlock(d22, N1, N1, s2.LFT.D22)
		if N1 > 0 {
			setBlock(d22, N1, 0, mulDense(s2.LFT.D21, s1.LFT.D12))
		}
	}

	b2 = resizeDense(b2, n, N)
	c2 = resizeDense(c2, N, n)
	d12 = resizeDense(d12, p, N)
	d21 = resizeDense(d21, N, m)

	sys, err := newNoCopy(A, B, C, D, s1.Dt)
	if err != nil {
		return nil, err
	}
	sys.LFT = &LFTDelay{
		Tau: tau,
		B2:  b2,
		C2:  c2,
		D12: d12,
		D21: d21,
		D22: d22,
	}
	sys.InputDelay = savedInput1
	sys.OutputDelay = savedOutput2
	sys.InputName = copyStringSlice(sys1.InputName)
	sys.OutputName = copyStringSlice(sys2.OutputName)
	sys.StateName = concatStringSlices([][]string{sys1.StateName, sys2.StateName}, []int{n1, n2})
	return sys, nil
}

func parallelLFT(sys1, sys2 *System) (*System, error) {
	_, m1, p1 := sys1.Dims()
	commonIn := mergeParallelDelay(sys1.InputDelay, sys2.InputDelay, m1)
	commonOut := mergeParallelDelay(sys1.OutputDelay, sys2.OutputDelay, p1)

	s1Stripped := sys1.Copy()
	s2Stripped := sys2.Copy()
	if commonIn != nil {
		if s1Stripped.InputDelay == nil {
			s1Stripped.InputDelay = make([]float64, m1)
		}
		if s2Stripped.InputDelay == nil {
			s2Stripped.InputDelay = make([]float64, m1)
		}
		for j := 0; j < m1; j++ {
			s1Stripped.InputDelay[j] -= commonIn[j]
			s2Stripped.InputDelay[j] -= commonIn[j]
		}
	}
	if commonOut != nil {
		if s1Stripped.OutputDelay == nil {
			s1Stripped.OutputDelay = make([]float64, p1)
		}
		if s2Stripped.OutputDelay == nil {
			s2Stripped.OutputDelay = make([]float64, p1)
		}
		for i := 0; i < p1; i++ {
			s1Stripped.OutputDelay[i] -= commonOut[i]
			s2Stripped.OutputDelay[i] -= commonOut[i]
		}
	}

	s1, err := s1Stripped.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}
	s2, err := s2Stripped.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}

	n1, _, _ := s1.Dims()
	n2, _, _ := s2.Dims()
	N1 := s1.internalDelayCount()
	N2 := s2.internalDelayCount()
	N := N1 + N2
	n := n1 + n2
	m := m1
	p := p1

	tau := make([]float64, 0, N)
	if s1.LFT != nil {
		tau = append(tau, s1.LFT.Tau...)
	}
	if s2.LFT != nil {
		tau = append(tau, s2.LFT.Tau...)
	}

	A := mat.NewDense(max(n, 1), max(n, 1), nil)
	B := mat.NewDense(max(n, 1), max(m, 1), nil)
	C := mat.NewDense(max(p, 1), max(n, 1), nil)
	D := mat.NewDense(max(p, 1), max(m, 1), nil)

	if n1 > 0 {
		setBlock(A, 0, 0, s1.A)
		setBlock(B, 0, 0, s1.B)
		setBlock(C, 0, 0, s1.C)
	}
	if n2 > 0 {
		setBlock(A, n1, n1, s2.A)
		setBlock(B, n1, 0, s2.B)
		setBlock(C, 0, n1, s2.C)
	}
	D.Add(s1.D, s2.D)

	A = resizeDense(A, n, n)
	B = resizeDense(B, n, m)
	C = resizeDense(C, p, n)
	D = resizeDense(D, p, m)

	if N == 0 {
		sys, err := newNoCopy(A, B, C, D, s1.Dt)
		if err != nil {
			return nil, err
		}
		sys.InputDelay = commonIn
		sys.OutputDelay = commonOut
		sys.InputName = copyStringSlice(sys1.InputName)
		sys.OutputName = copyStringSlice(sys1.OutputName)
		sys.StateName = concatStringSlices([][]string{sys1.StateName, sys2.StateName}, []int{n1, n2})
		return sys, nil
	}

	b2 := mat.NewDense(max(n, 1), N, nil)
	c2 := mat.NewDense(N, max(n, 1), nil)
	d12 := mat.NewDense(max(p, 1), N, nil)
	d21 := mat.NewDense(N, max(m, 1), nil)
	d22 := mat.NewDense(N, N, nil)

	if N1 > 0 {
		if n1 > 0 {
			setBlock(b2, 0, 0, s1.LFT.B2)
		}
		if n1 > 0 {
			setBlock(c2, 0, 0, s1.LFT.C2)
		}
		setBlock(d12, 0, 0, s1.LFT.D12)
		setBlock(d21, 0, 0, s1.LFT.D21)
		setBlock(d22, 0, 0, s1.LFT.D22)
	}

	if N2 > 0 {
		if n2 > 0 {
			setBlock(b2, n1, N1, s2.LFT.B2)
		}
		if n2 > 0 {
			setBlock(c2, N1, n1, s2.LFT.C2)
		}
		setBlock(d12, 0, N1, s2.LFT.D12)
		setBlock(d21, N1, 0, s2.LFT.D21)
		setBlock(d22, N1, N1, s2.LFT.D22)
	}

	b2 = resizeDense(b2, n, N)
	c2 = resizeDense(c2, N, n)
	d12 = resizeDense(d12, p, N)
	d21 = resizeDense(d21, N, m)

	sys, err := newNoCopy(A, B, C, D, s1.Dt)
	if err != nil {
		return nil, err
	}
	sys.LFT = &LFTDelay{
		Tau: tau,
		B2:  b2,
		C2:  c2,
		D12: d12,
		D21: d21,
		D22: d22,
	}
	sys.InputDelay = commonIn
	sys.OutputDelay = commonOut
	sys.InputName = copyStringSlice(sys1.InputName)
	sys.OutputName = copyStringSlice(sys1.OutputName)
	sys.StateName = concatStringSlices([][]string{sys1.StateName, sys2.StateName}, []int{n1, n2})
	return sys, nil
}

func appendInternalDelay(sys, sys1, sys2 *System, n1, n2, m1, m2, p1, p2 int) {
	N1 := sys1.internalDelayCount()
	N2 := sys2.internalDelayCount()
	if N1 == 0 && N2 == 0 {
		return
	}

	n := n1 + n2
	m := m1 + m2
	p := p1 + p2
	N := N1 + N2

	tau := make([]float64, N)
	if N1 > 0 {
		copy(tau, sys1.LFT.Tau)
	}
	if N2 > 0 {
		copy(tau[N1:], sys2.LFT.Tau)
	}

	b2 := mat.NewDense(max(n, 1), N, nil)
	c2 := mat.NewDense(N, max(n, 1), nil)
	d12 := mat.NewDense(max(p, 1), N, nil)
	d21 := mat.NewDense(N, max(m, 1), nil)
	d22 := mat.NewDense(N, N, nil)

	if N1 > 0 && sys1.LFT.B2 != nil {
		setBlock(b2, 0, 0, sys1.LFT.B2)
	}
	if N2 > 0 && sys2.LFT.B2 != nil {
		setBlock(b2, n1, N1, sys2.LFT.B2)
	}
	if N1 > 0 && sys1.LFT.C2 != nil {
		setBlock(c2, 0, 0, sys1.LFT.C2)
	}
	if N2 > 0 && sys2.LFT.C2 != nil {
		setBlock(c2, N1, n1, sys2.LFT.C2)
	}
	if N1 > 0 && sys1.LFT.D12 != nil {
		setBlock(d12, 0, 0, sys1.LFT.D12)
	}
	if N2 > 0 && sys2.LFT.D12 != nil {
		setBlock(d12, p1, N1, sys2.LFT.D12)
	}
	if N1 > 0 && sys1.LFT.D21 != nil {
		setBlock(d21, 0, 0, sys1.LFT.D21)
	}
	if N2 > 0 && sys2.LFT.D21 != nil {
		setBlock(d21, N1, m1, sys2.LFT.D21)
	}
	if N1 > 0 && sys1.LFT.D22 != nil {
		setBlock(d22, 0, 0, sys1.LFT.D22)
	}
	if N2 > 0 && sys2.LFT.D22 != nil {
		setBlock(d22, N1, N1, sys2.LFT.D22)
	}

	sys.LFT = &LFTDelay{
		Tau: tau,
		B2:  resizeDense(b2, n, N),
		C2:  resizeDense(c2, N, n),
		D12: resizeDense(d12, p, N),
		D21: resizeDense(d21, N, m),
		D22: resizeDense(d22, N, N),
	}
}


func BlkDiag(systems ...*System) (*System, error) {
	if len(systems) == 0 {
		return nil, fmt.Errorf("blkdiag: no systems provided: %w", ErrDimensionMismatch)
	}
	if len(systems) == 1 {
		return systems[0].Copy(), nil
	}

	dt := systems[0].Dt
	for i := 1; i < len(systems); i++ {
		if systems[i].Dt != dt {
			return nil, fmt.Errorf("blkdiag: system %d Dt=%v != Dt=%v: %w", i, systems[i].Dt, dt, ErrDomainMismatch)
		}
	}

	anyInternalDelay := false
	for _, s := range systems {
		if s.HasInternalDelay() {
			anyInternalDelay = true
			break
		}
	}

	srcs := systems
	if anyInternalDelay {
		srcs = make([]*System, len(systems))
		for i, s := range systems {
			pulled, err := s.PullDelaysToLFT()
			if err != nil {
				return nil, err
			}
			srcs[i] = pulled
		}
	}

	k := len(srcs)
	ns := make([]int, k)
	ms := make([]int, k)
	ps := make([]int, k)
	var nTotal, mTotal, pTotal int
	for i, s := range srcs {
		ns[i], ms[i], ps[i] = s.Dims()
		nTotal += ns[i]
		mTotal += ms[i]
		pTotal += ps[i]
	}

	A := mat.NewDense(max(nTotal, 1), max(nTotal, 1), nil)
	B := mat.NewDense(max(nTotal, 1), max(mTotal, 1), nil)
	C := mat.NewDense(max(pTotal, 1), max(nTotal, 1), nil)
	D := mat.NewDense(max(pTotal, 1), max(mTotal, 1), nil)

	nOff, mOff, pOff := 0, 0, 0
	for i, s := range srcs {
		if ns[i] > 0 {
			setBlock(A, nOff, nOff, s.A)
			setBlock(B, nOff, mOff, s.B)
			setBlock(C, pOff, nOff, s.C)
		}
		setBlock(D, pOff, mOff, s.D)
		nOff += ns[i]
		mOff += ms[i]
		pOff += ps[i]
	}

	A = resizeDense(A, nTotal, nTotal)
	B = resizeDense(B, nTotal, mTotal)
	C = resizeDense(C, pTotal, nTotal)
	D = resizeDense(D, pTotal, mTotal)

	var delay *mat.Dense
	hasIODelay := false
	for _, s := range systems {
		if s.Delay != nil {
			hasIODelay = true
			break
		}
	}
	if hasIODelay {
		delay = mat.NewDense(pTotal, mTotal, nil)
		pOff, mOff = 0, 0
		for i, s := range systems {
			if s.Delay != nil {
				setBlock(delay, pOff, mOff, s.Delay)
			}
			pOff += ps[i]
			mOff += ms[i]
		}
	}

	var inDel []float64
	hasIn := false
	for _, s := range systems {
		if s.InputDelay != nil {
			hasIn = true
			break
		}
	}
	if hasIn {
		inDel = make([]float64, mTotal)
		off := 0
		for i, s := range systems {
			if s.InputDelay != nil {
				copy(inDel[off:], s.InputDelay)
			}
			off += ms[i]
		}
	}

	var outDel []float64
	hasOut := false
	for _, s := range systems {
		if s.OutputDelay != nil {
			hasOut = true
			break
		}
	}
	if hasOut {
		outDel = make([]float64, pTotal)
		off := 0
		for i, s := range systems {
			if s.OutputDelay != nil {
				copy(outDel[off:], s.OutputDelay)
			}
			off += ps[i]
		}
	}

	var inNames, outNames, stNames [][]string
	for _, s := range systems {
		inNames = append(inNames, s.InputName)
		outNames = append(outNames, s.OutputName)
		stNames = append(stNames, s.StateName)
	}

	if nTotal == 0 {
		sys, _ := NewGain(D, dt)
		sys.Delay = delay
		sys.InputDelay = inDel
		sys.OutputDelay = outDel
		blkDiagInternalDelay(sys, srcs, ns, ms, ps, nTotal, mTotal, pTotal)
		sys.InputName = concatStringSlices(inNames, ms)
		sys.OutputName = concatStringSlices(outNames, ps)
		sys.StateName = concatStringSlices(stNames, ns)
		return sys, nil
	}

	sys, err := newNoCopy(A, B, C, D, dt)
	if err != nil {
		return nil, err
	}
	sys.Delay = delay
	sys.InputDelay = inDel
	sys.OutputDelay = outDel
	blkDiagInternalDelay(sys, srcs, ns, ms, ps, nTotal, mTotal, pTotal)
	sys.InputName = concatStringSlices(inNames, ms)
	sys.OutputName = concatStringSlices(outNames, ps)
	sys.StateName = concatStringSlices(stNames, ns)
	return sys, nil
}

func blkDiagInternalDelay(sys *System, srcs []*System, ns, ms, ps []int, nTotal, mTotal, pTotal int) {
	var NTotal int
	for _, s := range srcs {
		NTotal += s.internalDelayCount()
	}
	if NTotal == 0 {
		return
	}

	tau := make([]float64, 0, NTotal)
	for _, s := range srcs {
		if s.LFT != nil {
			tau = append(tau, s.LFT.Tau...)
		}
	}

	n := nTotal
	m := mTotal
	p := pTotal

	b2 := mat.NewDense(max(n, 1), NTotal, nil)
	c2 := mat.NewDense(NTotal, max(n, 1), nil)
	d12 := mat.NewDense(max(p, 1), NTotal, nil)
	d21 := mat.NewDense(NTotal, max(m, 1), nil)
	d22 := mat.NewDense(NTotal, NTotal, nil)

	nOff, mOff, pOff, NOff := 0, 0, 0, 0
	for i, s := range srcs {
		Ni := s.internalDelayCount()
		if Ni > 0 {
			if ns[i] > 0 && s.LFT.B2 != nil {
				setBlock(b2, nOff, NOff, s.LFT.B2)
			}
			if ns[i] > 0 && s.LFT.C2 != nil {
				setBlock(c2, NOff, nOff, s.LFT.C2)
			}
			if s.LFT.D12 != nil {
				setBlock(d12, pOff, NOff, s.LFT.D12)
			}
			if s.LFT.D21 != nil {
				setBlock(d21, NOff, mOff, s.LFT.D21)
			}
			if s.LFT.D22 != nil {
				setBlock(d22, NOff, NOff, s.LFT.D22)
			}
		}
		nOff += ns[i]
		mOff += ms[i]
		pOff += ps[i]
		NOff += Ni
	}

	sys.LFT = &LFTDelay{
		Tau: tau,
		B2:  resizeDense(b2, n, NTotal),
		C2:  resizeDense(c2, NTotal, n),
		D12: resizeDense(d12, p, NTotal),
		D21: resizeDense(d21, NTotal, m),
		D22: resizeDense(d22, NTotal, NTotal),
	}
}

func Connect(sys *System, Q *mat.Dense, inputs, outputs []int) (*System, error) {
	if sys == nil {
		return nil, fmt.Errorf("connect: system cannot be nil")
	}
	n, m, p := sys.Dims()

	qr, qc := Q.Dims()
	if qr != m || qc != p {
		return nil, fmt.Errorf("connect: Q size %dx%d != %dx%d: %w", qr, qc, m, p, ErrDimensionMismatch)
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("connect: inputs must be non-empty: %w", ErrDimensionMismatch)
	}
	if len(outputs) == 0 {
		return nil, fmt.Errorf("connect: outputs must be non-empty: %w", ErrDimensionMismatch)
	}

	inSeen := make(map[int]bool, len(inputs))
	for _, idx := range inputs {
		if idx < 0 || idx >= m {
			return nil, fmt.Errorf("connect: input index %d out of range [0,%d): %w", idx, m, ErrDimensionMismatch)
		}
		if inSeen[idx] {
			return nil, fmt.Errorf("connect: duplicate input index %d: %w", idx, ErrDimensionMismatch)
		}
		inSeen[idx] = true
	}
	outSeen := make(map[int]bool, len(outputs))
	for _, idx := range outputs {
		if idx < 0 || idx >= p {
			return nil, fmt.Errorf("connect: output index %d out of range [0,%d): %w", idx, p, ErrDimensionMismatch)
		}
		if outSeen[idx] {
			return nil, fmt.Errorf("connect: duplicate output index %d: %w", idx, ErrDimensionMismatch)
		}
		outSeen[idx] = true
	}

	if sys.HasDelay() || sys.HasInternalDelay() {
		return connectWithDelay(sys, Q, inputs, outputs)
	}

	return connectSimple(sys, Q, inputs, outputs, n, m, p)
}

func connectWithDelay(sys *System, Q *mat.Dense, inputs, outputs []int) (*System, error) {
	_, m, p := sys.Dims()

	savedInputDelay := make([]float64, len(inputs))
	if sys.InputDelay != nil {
		for k, j := range inputs {
			savedInputDelay[k] = sys.InputDelay[j]
		}
	}
	savedOutputDelay := make([]float64, len(outputs))
	if sys.OutputDelay != nil {
		for k, i := range outputs {
			savedOutputDelay[k] = sys.OutputDelay[i]
		}
	}

	sCopy := sys.Copy()
	if sCopy.InputDelay != nil {
		for _, j := range inputs {
			sCopy.InputDelay[j] = 0
		}
	}
	if sCopy.OutputDelay != nil {
		for _, i := range outputs {
			sCopy.OutputDelay[i] = 0
		}
	}

	sLFT, err := sCopy.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}

	N := sLFT.internalDelayCount()
	H, tau := sLFT.GetDelayModel()

	nH, _, _ := H.Dims()

	Dcore := extractBlock(H.D, 0, 0, p, m)
	QD := mat.NewDense(m, m, nil)
	QD.Mul(Q, Dcore)
	IQD := eyeDense(m)
	IQD.Sub(IQD, QD)

	var lu mat.LU
	lu.Factorize(IQD)
	if lu.Det() == 0 {
		return nil, fmt.Errorf("connect: (I-Q*D) singular, algebraic loop: %w", ErrAlgebraicLoop)
	}

	eyeM := eyeDense(m)
	E := mat.NewDense(m, m, nil)
	if err := lu.SolveTo(E, false, eyeM); err != nil {
		return nil, fmt.Errorf("connect: LU solve failed: %w", ErrSingularTransform)
	}

	EQ := mat.NewDense(m, p, nil)
	EQ.Mul(E, Q)

	DEQ := mat.NewDense(p, p, nil)
	DEQ.Mul(Dcore, EQ)
	Er := eyeDense(p)
	Er.Add(Er, DEQ)

	Bcore := extractBlock(H.B, 0, 0, nH, m)
	Ccore := extractBlock(H.C, 0, 0, p, nH)

	mExt := len(inputs)
	pExt := len(outputs)
	mTotal := mExt + N
	pTotal := pExt + N

	var BEQ *mat.Dense
	if nH > 0 {
		BEQ = mat.NewDense(nH, p, nil)
		BEQ.Mul(Bcore, EQ)
	}

	var Dout, Din, DinEQ *mat.Dense
	if N > 0 {
		Dout = extractBlock(H.D, 0, m, p, N)
		Din = extractBlock(H.D, p, 0, N, m)
		DinEQ = mat.NewDense(N, p, nil)
		DinEQ.Mul(Din, EQ)
	}

	Acl := mat.NewDense(max(nH, 1), max(nH, 1), nil)
	if nH > 0 {
		setBlock(Acl, 0, 0, H.A)
		BEQC := mat.NewDense(nH, nH, nil)
		BEQC.Mul(BEQ, Ccore)
		addBlock(Acl, 0, 0, BEQC)
	}

	Bcl := mat.NewDense(max(nH, 1), max(mTotal, 1), nil)
	if nH > 0 {
		Bfull := mat.NewDense(nH, m, nil)
		Bfull.Mul(Bcore, E)
		bfRaw := Bfull.RawMatrix()
		for k, j := range inputs {
			for i := 0; i < nH; i++ {
				Bcl.Set(i, k, bfRaw.Data[i*bfRaw.Stride+j])
			}
		}
		if N > 0 {
			Bdelay := extractBlock(H.B, 0, m, nH, N)
			BEQDout := mat.NewDense(nH, N, nil)
			BEQDout.Mul(BEQ, Dout)
			b2mod := mat.NewDense(nH, N, nil)
			b2mod.Add(Bdelay, BEQDout)
			setBlock(Bcl, 0, mExt, b2mod)
		}
	}

	Cfull := mat.NewDense(p, max(nH, 1), nil)
	if nH > 0 {
		Cfull.Mul(Er, Ccore)
	}
	Dfull := mat.NewDense(p, m, nil)
	Dfull.Mul(Er, Dcore)

	Ccl := mat.NewDense(max(pTotal, 1), max(nH, 1), nil)
	if nH > 0 {
		cfRaw := Cfull.RawMatrix()
		ccRaw := Ccl.RawMatrix()
		for k, i := range outputs {
			copy(ccRaw.Data[k*ccRaw.Stride:k*ccRaw.Stride+nH],
				cfRaw.Data[i*cfRaw.Stride:i*cfRaw.Stride+nH])
		}
		if N > 0 {
			Cdelay := extractBlock(H.C, p, 0, N, nH)
			DinEQC := mat.NewDense(N, nH, nil)
			DinEQC.Mul(DinEQ, Ccore)
			c2mod := mat.NewDense(N, nH, nil)
			c2mod.Add(Cdelay, DinEQC)
			setBlock(Ccl, pExt, 0, c2mod)
		}
	}

	Dcl := mat.NewDense(max(pTotal, 1), max(mTotal, 1), nil)
	for ki, oi := range outputs {
		for kj, ij := range inputs {
			Dcl.Set(ki, kj, Dfull.At(oi, ij))
		}
	}

	if N > 0 {
		ErDout := mat.NewDense(p, N, nil)
		ErDout.Mul(Er, Dout)
		erRaw := ErDout.RawMatrix()
		for ki, oi := range outputs {
			for j := 0; j < N; j++ {
				Dcl.Set(ki, mExt+j, erRaw.Data[oi*erRaw.Stride+j])
			}
		}

		DinE := mat.NewDense(N, m, nil)
		DinE.Mul(Din, E)
		deRaw := DinE.RawMatrix()
		for i := 0; i < N; i++ {
			for kj, ij := range inputs {
				Dcl.Set(pExt+i, kj, deRaw.Data[i*deRaw.Stride+ij])
			}
		}

		D22 := extractBlock(H.D, p, m, N, N)
		d22cross := mat.NewDense(N, N, nil)
		d22cross.Mul(DinEQ, Dout)
		d22mod := mat.NewDense(N, N, nil)
		d22mod.Add(D22, d22cross)
		setBlock(Dcl, pExt, mExt, d22mod)
	}

	Acl = resizeDense(Acl, nH, nH)
	Bcl = resizeDense(Bcl, nH, mTotal)
	Ccl = resizeDense(Ccl, pTotal, nH)
	Dcl = resizeDense(Dcl, pTotal, mTotal)

	if nH == 0 {
		Acl = &mat.Dense{}
	}

	Hresult := &System{A: Acl, B: Bcl, C: Ccl, D: Dcl, Dt: sys.Dt}
	result, err := SetDelayModel(Hresult, tau)
	if err != nil {
		return nil, err
	}

	hasExtIn := false
	for _, v := range savedInputDelay {
		if v != 0 {
			hasExtIn = true
			break
		}
	}
	if hasExtIn {
		result.InputDelay = savedInputDelay
	}
	hasExtOut := false
	for _, v := range savedOutputDelay {
		if v != 0 {
			hasExtOut = true
			break
		}
	}
	if hasExtOut {
		result.OutputDelay = savedOutputDelay
	}

	result.InputName = selectStringSlice(sys.InputName, inputs)
	result.OutputName = selectStringSlice(sys.OutputName, outputs)

	return result, nil
}

func connectSimple(sys *System, Q *mat.Dense, inputs, outputs []int, n, m, p int) (*System, error) {
	mExt := len(inputs)
	pExt := len(outputs)

	QD := mat.NewDense(m, m, nil)
	QD.Mul(Q, sys.D)
	IQD := eyeDense(m)
	IQD.Sub(IQD, QD)

	var lu mat.LU
	lu.Factorize(IQD)
	if lu.Det() == 0 {
		return nil, fmt.Errorf("connect: (I-Q*D) singular, algebraic loop: %w", ErrAlgebraicLoop)
	}

	eyeM := eyeDense(m)
	E := mat.NewDense(m, m, nil)
	if err := lu.SolveTo(E, false, eyeM); err != nil {
		return nil, fmt.Errorf("connect: LU solve failed: %w", ErrSingularTransform)
	}

	EQ := mat.NewDense(m, p, nil)
	EQ.Mul(E, Q)

	if n == 0 {
		DEQ := mat.NewDense(p, p, nil)
		DEQ.Mul(sys.D, EQ)
		Er := eyeDense(p)
		Er.Add(Er, DEQ)

		Dfull := mat.NewDense(p, m, nil)
		Dfull.Mul(Er, sys.D)

		Dcl := mat.NewDense(pExt, mExt, nil)
		for ki, oi := range outputs {
			for kj, ij := range inputs {
				Dcl.Set(ki, kj, Dfull.At(oi, ij))
			}
		}
		result, err := NewGain(Dcl, sys.Dt)
		if err != nil {
			return nil, err
		}
		result.InputName = selectStringSlice(sys.InputName, inputs)
		result.OutputName = selectStringSlice(sys.OutputName, outputs)
		result.StateName = copyStringSlice(sys.StateName)
		return result, nil
	}

	BEQC := mat.NewDense(n, n, nil)
	BEQ := mat.NewDense(n, p, nil)
	BEQ.Mul(sys.B, EQ)
	BEQC.Mul(BEQ, sys.C)
	Acl := mat.NewDense(n, n, nil)
	Acl.Add(sys.A, BEQC)

	Bfull := mat.NewDense(n, m, nil)
	Bfull.Mul(sys.B, E)

	DEQ := mat.NewDense(p, p, nil)
	DEQ.Mul(sys.D, EQ)
	Er := eyeDense(p)
	Er.Add(Er, DEQ)

	Cfull := mat.NewDense(p, n, nil)
	Cfull.Mul(Er, sys.C)

	Dfull := mat.NewDense(p, m, nil)
	Dfull.Mul(Er, sys.D)

	Bcl := mat.NewDense(n, mExt, nil)
	bfRaw := Bfull.RawMatrix()
	bcRaw := Bcl.RawMatrix()
	for k, j := range inputs {
		for i := 0; i < n; i++ {
			bcRaw.Data[i*bcRaw.Stride+k] = bfRaw.Data[i*bfRaw.Stride+j]
		}
	}

	Ccl := mat.NewDense(pExt, n, nil)
	cfRaw := Cfull.RawMatrix()
	ccRaw := Ccl.RawMatrix()
	for k, i := range outputs {
		copy(ccRaw.Data[k*ccRaw.Stride:k*ccRaw.Stride+n], cfRaw.Data[i*cfRaw.Stride:i*cfRaw.Stride+n])
	}

	Dcl := mat.NewDense(pExt, mExt, nil)
	for ki, oi := range outputs {
		for kj, ij := range inputs {
			Dcl.Set(ki, kj, Dfull.At(oi, ij))
		}
	}

	result, err := newNoCopy(Acl, Bcl, Ccl, Dcl, sys.Dt)
	if err != nil {
		return nil, err
	}
	result.InputName = selectStringSlice(sys.InputName, inputs)
	result.OutputName = selectStringSlice(sys.OutputName, outputs)
	result.StateName = copyStringSlice(sys.StateName)
	return result, nil
}

func buildSystem(A, B, C, D *mat.Dense, dt float64, delay *mat.Dense) (*System, error) {
	if A == nil && B == nil && C == nil {
		sys, err := NewGain(D, dt)
		if err != nil {
			return nil, err
		}
		sys.Delay = delay
		return sys, nil
	}
	sys, err := newNoCopy(A, B, C, D, dt)
	if err != nil {
		return nil, err
	}
	sys.Delay = delay
	return sys, nil
}
