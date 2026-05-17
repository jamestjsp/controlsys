package controlsys

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

const delayTopologyTol = 1e-12

type delayTopology struct {
	sys *System
	p   int
	m   int
}

func newDelayTopology(sys *System) delayTopology {
	_, m, p := sys.Dims()
	return delayTopology{sys: sys, p: p, m: m}
}

func (dt delayTopology) total(includeDelayMatrix bool) *mat.Dense {
	return effectiveIODelayMatrix(dt.sys, dt.p, dt.m, includeDelayMatrix)
}

func (dt delayTopology) decomposeTotal() (inputDelay, outputDelay []float64, residual *mat.Dense) {
	total := dt.total(true)
	if total == nil {
		return nil, nil, nil
	}
	return DecomposeIODelay(total)
}

func delayMatrixHasNonzero(m *mat.Dense) bool {
	return delayMatrixHasNonzeroTol(m, 0)
}

func delayMatrixHasNonzeroTol(m *mat.Dense, tol float64) bool {
	if m == nil {
		return false
	}
	raw := m.RawMatrix()
	for i := 0; i < raw.Rows; i++ {
		for j := 0; j < raw.Cols; j++ {
			if math.Abs(raw.Data[i*raw.Stride+j]) > tol {
				return true
			}
		}
	}
	return false
}

func delaySliceHasNonzero(s []float64) bool {
	for _, v := range s {
		if v != 0 {
			return true
		}
	}
	return false
}
