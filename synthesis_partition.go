package controlsys

import "gonum.org/v1/gonum/mat"

type generalizedPlantPartition struct {
	A                              *mat.Dense
	B1, B2                         *mat.Dense
	C1, C2                         *mat.Dense
	D11, D12, D21, D22             *mat.Dense
	n, m, p, m1, m2, p1, p2        int
	measurementNames, controlNames []string
}

func partitionGeneralizedPlant(P *System, nmeas, ncont int) (*generalizedPlantPartition, error) {
	if !P.IsContinuous() {
		return nil, ErrWrongDomain
	}
	if P.IsDescriptor() {
		return nil, ErrDescriptorRiccati
	}

	n, m, p := P.Dims()
	if n == 0 {
		return nil, ErrInvalidPartition
	}
	if ncont <= 0 || nmeas <= 0 || ncont > m || nmeas > p {
		return nil, ErrInvalidPartition
	}

	m1 := m - ncont
	m2 := ncont
	p1 := p - nmeas
	p2 := nmeas
	if m1 <= 0 || p1 <= 0 {
		return nil, ErrInvalidPartition
	}

	return &generalizedPlantPartition{
		A:                P.A,
		B1:               extractBlock(P.B, 0, 0, n, m1),
		B2:               extractBlock(P.B, 0, m1, n, m2),
		C1:               extractBlock(P.C, 0, 0, p1, n),
		C2:               extractBlock(P.C, p1, 0, p2, n),
		D11:              extractBlock(P.D, 0, 0, p1, m1),
		D12:              extractBlock(P.D, 0, m1, p1, m2),
		D21:              extractBlock(P.D, p1, 0, p2, m1),
		D22:              extractBlock(P.D, p1, m1, p2, m2),
		n:                n,
		m:                m,
		p:                p,
		m1:               m1,
		m2:               m2,
		p1:               p1,
		p2:               p2,
		measurementNames: selectTrailingNames(P.OutputName, p1, p2),
		controlNames:     selectTrailingNames(P.InputName, m1, m2),
	}, nil
}

func (gp *generalizedPlantPartition) applyControllerNames(K *System) {
	K.InputName = copyStringSlice(gp.measurementNames)
	K.OutputName = copyStringSlice(gp.controlNames)
}

func selectTrailingNames(names []string, start, count int) []string {
	if names == nil {
		return nil
	}
	return selectStringSlice(names, contiguousIndices(start, count))
}

func contiguousIndices(start, count int) []int {
	idx := make([]int, count)
	for i := range idx {
		idx[i] = start + i
	}
	return idx
}
