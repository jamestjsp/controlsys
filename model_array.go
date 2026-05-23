package controlsys

import "fmt"

type ModelArray struct {
	models     []*System
	shape      []int
	n          int
	m          int
	p          int
	dt         float64
	inputName  []string
	outputName []string
}

type ModelArrayFreqResponse struct {
	Shape     []int
	Responses []*FreqResponseMatrix
	Void      []bool
	Omega     []float64
	P         int
	M         int
}

type ModelArrayTimeResponse struct {
	Shape     []int
	Responses []*TimeResponse
	Void      []bool
}

func NewModelArray(shape []int, models []*System) (*ModelArray, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("NewModelArray: shape is empty: %w", ErrDimensionMismatch)
	}
	total := 1
	for _, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("NewModelArray: invalid shape dimension %d: %w", dim, ErrDimensionMismatch)
		}
		total *= dim
	}
	if total != len(models) {
		return nil, fmt.Errorf("NewModelArray: shape product %d != %d models: %w", total, len(models), ErrDimensionMismatch)
	}

	arr := &ModelArray{
		models: make([]*System, len(models)),
		shape:  copyIntSlice(shape),
	}
	var ref *System
	for _, sys := range models {
		if sys == nil {
			continue
		}
		if err := sys.Validate(); err != nil {
			return nil, fmt.Errorf("NewModelArray: %w", err)
		}
		if ref == nil {
			ref = sys
			arr.n, arr.m, arr.p = sys.Dims()
			arr.dt = sys.Dt
			arr.inputName = copyStringSlice(sys.InputName)
			arr.outputName = copyStringSlice(sys.OutputName)
			continue
		}
		if err := validateModelArrayCompatible(ref, sys); err != nil {
			return nil, fmt.Errorf("NewModelArray: %w", err)
		}
	}
	for i, sys := range models {
		if sys != nil {
			arr.models[i] = sys.Copy()
		}
	}
	return arr, nil
}

func StackModelArrays(arrays ...*ModelArray) (*ModelArray, error) {
	if len(arrays) == 0 {
		return nil, fmt.Errorf("StackModelArrays: no arrays: %w", ErrDimensionMismatch)
	}
	var ref *ModelArray
	total := 0
	for _, arr := range arrays {
		if arr == nil {
			return nil, fmt.Errorf("StackModelArrays: nil array: %w", ErrDimensionMismatch)
		}
		if ref == nil {
			ref = arr
		} else if err := validateModelArrayHeadersCompatible(ref, arr); err != nil {
			return nil, fmt.Errorf("StackModelArrays: %w", err)
		}
		total += arr.Len()
	}
	models := make([]*System, 0, total)
	for _, arr := range arrays {
		for _, sys := range arr.models {
			if sys == nil {
				models = append(models, nil)
			} else {
				models = append(models, sys.Copy())
			}
		}
	}
	return NewModelArray([]int{total}, models)
}

func (a *ModelArray) Shape() []int {
	if a == nil {
		return nil
	}
	return copyIntSlice(a.shape)
}

func (a *ModelArray) Len() int {
	if a == nil {
		return 0
	}
	return len(a.models)
}

func (a *ModelArray) Dims() (n, m, p int) {
	if a == nil {
		return 0, 0, 0
	}
	return a.n, a.m, a.p
}

func (a *ModelArray) InputName() []string {
	if a == nil {
		return nil
	}
	return copyStringSlice(a.inputName)
}

func (a *ModelArray) OutputName() []string {
	if a == nil {
		return nil
	}
	return copyStringSlice(a.outputName)
}

func (a *ModelArray) Model(index ...int) (*System, bool, error) {
	if a == nil {
		return nil, false, fmt.Errorf("ModelArray.Model: nil array: %w", ErrDimensionMismatch)
	}
	flat, err := a.flatIndex(index)
	if err != nil {
		return nil, false, err
	}
	return a.ModelFlat(flat)
}

func (a *ModelArray) ModelFlat(index int) (*System, bool, error) {
	if a == nil {
		return nil, false, fmt.Errorf("ModelArray.ModelFlat: nil array: %w", ErrDimensionMismatch)
	}
	if index < 0 || index >= len(a.models) {
		return nil, false, fmt.Errorf("ModelArray.ModelFlat: index %d out of range: %w", index, ErrDimensionMismatch)
	}
	sys := a.models[index]
	if sys == nil {
		return nil, false, nil
	}
	return sys.Copy(), true, nil
}

func (a *ModelArray) SelectFlat(indices ...int) (*ModelArray, error) {
	if a == nil {
		return nil, fmt.Errorf("ModelArray.SelectFlat: nil array: %w", ErrDimensionMismatch)
	}
	models := make([]*System, len(indices))
	for i, idx := range indices {
		if idx < 0 || idx >= len(a.models) {
			return nil, fmt.Errorf("ModelArray.SelectFlat: index %d out of range: %w", idx, ErrDimensionMismatch)
		}
		if a.models[idx] != nil {
			models[i] = a.models[idx].Copy()
		}
	}
	return NewModelArray([]int{len(indices)}, models)
}

func (a *ModelArray) FreqResponse(omega []float64) (*ModelArrayFreqResponse, error) {
	if a == nil {
		return nil, fmt.Errorf("ModelArray.FreqResponse: nil array: %w", ErrDimensionMismatch)
	}
	out := &ModelArrayFreqResponse{
		Shape:     copyIntSlice(a.shape),
		Responses: make([]*FreqResponseMatrix, len(a.models)),
		Void:      make([]bool, len(a.models)),
		Omega:     copyFloatSlice(omega),
		P:         a.p,
		M:         a.m,
	}
	for i, sys := range a.models {
		if sys == nil {
			out.Void[i] = true
			continue
		}
		resp, err := sys.FreqResponse(omega)
		if err != nil {
			return nil, fmt.Errorf("ModelArray.FreqResponse[%d]: %w", i, err)
		}
		out.Responses[i] = resp
	}
	return out, nil
}

func (a *ModelArray) Step(tFinal float64) (*ModelArrayTimeResponse, error) {
	if a == nil {
		return nil, fmt.Errorf("ModelArray.Step: nil array: %w", ErrDimensionMismatch)
	}
	out := &ModelArrayTimeResponse{
		Shape:     copyIntSlice(a.shape),
		Responses: make([]*TimeResponse, len(a.models)),
		Void:      make([]bool, len(a.models)),
	}
	for i, sys := range a.models {
		if sys == nil {
			out.Void[i] = true
			continue
		}
		resp, err := Step(sys, tFinal)
		if err != nil {
			return nil, fmt.Errorf("ModelArray.Step[%d]: %w", i, err)
		}
		out.Responses[i] = resp
	}
	return out, nil
}

func (a *ModelArray) flatIndex(index []int) (int, error) {
	if len(index) != len(a.shape) {
		return 0, fmt.Errorf("ModelArray.Model: got %d indices for rank %d: %w", len(index), len(a.shape), ErrDimensionMismatch)
	}
	flat := 0
	stride := 1
	for dim := len(a.shape) - 1; dim >= 0; dim-- {
		idx := index[dim]
		if idx < 0 || idx >= a.shape[dim] {
			return 0, fmt.Errorf("ModelArray.Model: index %d out of range for dimension %d: %w", idx, dim, ErrDimensionMismatch)
		}
		flat += idx * stride
		stride *= a.shape[dim]
	}
	return flat, nil
}

func validateModelArrayCompatible(ref, sys *System) error {
	_, rm, rp := ref.Dims()
	_, sm, sp := sys.Dims()
	if rm != sm || rp != sp {
		return fmt.Errorf("model dimensions (%d,%d) != (%d,%d): %w", sp, sm, rp, rm, ErrDimensionMismatch)
	}
	if ref.Dt != sys.Dt {
		return fmt.Errorf("sample time %g != %g: %w", sys.Dt, ref.Dt, ErrDimensionMismatch)
	}
	if !stringSlicesCompatible(ref.InputName, sys.InputName) || !stringSlicesCompatible(ref.OutputName, sys.OutputName) {
		return fmt.Errorf("signal names differ: %w", ErrDimensionMismatch)
	}
	return nil
}

func validateModelArrayHeadersCompatible(ref, arr *ModelArray) error {
	if ref.m != arr.m || ref.p != arr.p {
		return fmt.Errorf("model dimensions (%d,%d) != (%d,%d): %w", arr.p, arr.m, ref.p, ref.m, ErrDimensionMismatch)
	}
	if ref.dt != arr.dt {
		return fmt.Errorf("sample time %g != %g: %w", arr.dt, ref.dt, ErrDimensionMismatch)
	}
	if !stringSlicesCompatible(ref.inputName, arr.inputName) || !stringSlicesCompatible(ref.outputName, arr.outputName) {
		return fmt.Errorf("signal names differ: %w", ErrDimensionMismatch)
	}
	return nil
}

func stringSlicesCompatible(a, b []string) bool {
	if len(a) == 0 || len(b) == 0 {
		return true
	}
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func copyIntSlice(s []int) []int {
	if s == nil {
		return nil
	}
	out := make([]int, len(s))
	copy(out, s)
	return out
}
