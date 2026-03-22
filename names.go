package controlsys

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func copyStringSlice(s []string) []string {
	if s == nil {
		return nil
	}
	out := make([]string, len(s))
	copy(out, s)
	return out
}

func concatStringSlices(slices [][]string, lengths []int) []string {
	allNil := true
	for _, s := range slices {
		if s != nil {
			allNil = false
			break
		}
	}
	if allNil {
		return nil
	}
	total := 0
	for _, l := range lengths {
		total += l
	}
	out := make([]string, 0, total)
	for i, s := range slices {
		if s != nil {
			out = append(out, s...)
		} else {
			out = append(out, make([]string, lengths[i])...)
		}
	}
	return out
}

func selectStringSlice(names []string, indices []int) []string {
	if names == nil {
		return nil
	}
	out := make([]string, len(indices))
	for i, idx := range indices {
		if idx < len(names) {
			out[i] = names[idx]
		}
	}
	return out
}

func lookupSignalIndex(names []string, name string) (int, error) {
	for i, n := range names {
		if n == name {
			return i, nil
		}
	}
	return -1, fmt.Errorf("%q: %w", name, ErrSignalNotFound)
}

func lookupSignalIndices(names []string, targets []string) ([]int, error) {
	indices := make([]int, len(targets))
	for i, t := range targets {
		idx, err := lookupSignalIndex(names, t)
		if err != nil {
			return nil, err
		}
		indices[i] = idx
	}
	return indices, nil
}

func propagateNames(result, src *System) {
	result.InputName = copyStringSlice(src.InputName)
	result.OutputName = copyStringSlice(src.OutputName)
	result.StateName = copyStringSlice(src.StateName)
	result.Notes = src.Notes
}

func propagateIONames(result, src *System) {
	result.InputName = copyStringSlice(src.InputName)
	result.OutputName = copyStringSlice(src.OutputName)
}

func (sys *System) SelectByIndex(inputs, outputs []int) (*System, error) {
	n, m, p := sys.Dims()
	for _, idx := range inputs {
		if idx < 0 || idx >= m {
			return nil, fmt.Errorf("select: input index %d out of range [0,%d): %w", idx, m, ErrDimensionMismatch)
		}
	}
	for _, idx := range outputs {
		if idx < 0 || idx >= p {
			return nil, fmt.Errorf("select: output index %d out of range [0,%d): %w", idx, p, ErrDimensionMismatch)
		}
	}

	mSel := len(inputs)
	pSel := len(outputs)

	Bsel := mat.NewDense(max(n, 1), max(mSel, 1), nil)
	if n > 0 {
		bRaw := sys.B.RawMatrix()
		bsRaw := Bsel.RawMatrix()
		for k, j := range inputs {
			for i := 0; i < n; i++ {
				bsRaw.Data[i*bsRaw.Stride+k] = bRaw.Data[i*bRaw.Stride+j]
			}
		}
	}

	Csel := mat.NewDense(max(pSel, 1), max(n, 1), nil)
	if n > 0 {
		cRaw := sys.C.RawMatrix()
		csRaw := Csel.RawMatrix()
		for k, i := range outputs {
			copy(csRaw.Data[k*csRaw.Stride:k*csRaw.Stride+n], cRaw.Data[i*cRaw.Stride:i*cRaw.Stride+n])
		}
	}

	Dsel := mat.NewDense(pSel, mSel, nil)
	for ki, oi := range outputs {
		for kj, ij := range inputs {
			Dsel.Set(ki, kj, sys.D.At(oi, ij))
		}
	}

	if n == 0 {
		result, err := NewGain(Dsel, sys.Dt)
		if err != nil {
			return nil, err
		}
		result.InputName = selectStringSlice(sys.InputName, inputs)
		result.OutputName = selectStringSlice(sys.OutputName, outputs)
		return result, nil
	}

	Bsel = resizeDense(Bsel, n, mSel)
	Csel = resizeDense(Csel, pSel, n)

	result, err := newNoCopy(denseCopy(sys.A), Bsel, Csel, Dsel, sys.Dt)
	if err != nil {
		return nil, err
	}
	result.InputName = selectStringSlice(sys.InputName, inputs)
	result.OutputName = selectStringSlice(sys.OutputName, outputs)
	result.StateName = copyStringSlice(sys.StateName)

	if sys.InputDelay != nil {
		result.InputDelay = make([]float64, mSel)
		for k, j := range inputs {
			result.InputDelay[k] = sys.InputDelay[j]
		}
	}
	if sys.OutputDelay != nil {
		result.OutputDelay = make([]float64, pSel)
		for k, i := range outputs {
			result.OutputDelay[k] = sys.OutputDelay[i]
		}
	}

	return result, nil
}

func (sys *System) SelectByName(inputs, outputs []string) (*System, error) {
	inIdx, err := lookupSignalIndices(sys.InputName, inputs)
	if err != nil {
		return nil, fmt.Errorf("select inputs: %w", err)
	}
	outIdx, err := lookupSignalIndices(sys.OutputName, outputs)
	if err != nil {
		return nil, fmt.Errorf("select outputs: %w", err)
	}
	return sys.SelectByIndex(inIdx, outIdx)
}

type Connection struct {
	From string
	To   string
	Gain float64
}

func ConnectByName(systems []*System, connections []Connection, inputs, outputs []string) (*System, error) {
	aug, err := BlkDiag(systems...)
	if err != nil {
		return nil, fmt.Errorf("connectbyname: %w", err)
	}

	_, m, p := aug.Dims()

	inIdx, err := lookupSignalIndices(aug.InputName, inputs)
	if err != nil {
		return nil, fmt.Errorf("connectbyname inputs: %w", err)
	}
	outIdx, err := lookupSignalIndices(aug.OutputName, outputs)
	if err != nil {
		return nil, fmt.Errorf("connectbyname outputs: %w", err)
	}

	Q := mat.NewDense(m, p, nil)
	for _, c := range connections {
		fromIdx, err := lookupSignalIndex(aug.OutputName, c.From)
		if err != nil {
			return nil, fmt.Errorf("connectbyname connection from: %w", err)
		}
		toIdx, err := lookupSignalIndex(aug.InputName, c.To)
		if err != nil {
			return nil, fmt.Errorf("connectbyname connection to: %w", err)
		}
		gain := c.Gain
		if gain == 0 {
			gain = 1
		}
		Q.Set(toIdx, fromIdx, gain)
	}

	return Connect(aug, Q, inIdx, outIdx)
}
