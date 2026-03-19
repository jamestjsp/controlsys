package controlsys

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func denseCopyTo(dst, src *mat.Dense) {
	if src == nil {
		dRaw := dst.RawMatrix()
		for i := 0; i < dRaw.Rows; i++ {
			row := dRaw.Data[i*dRaw.Stride : i*dRaw.Stride+dRaw.Cols]
			for j := range row {
				row[j] = 0
			}
		}
		return
	}
	dst.Copy(src)
}

func denseCopy(m *mat.Dense) *mat.Dense {
	if m == nil {
		return &mat.Dense{}
	}
	r, c := m.Dims()
	if r == 0 || c == 0 {
		return &mat.Dense{}
	}
	return mat.DenseCopyOf(m)
}

func denseCopySafe(m *mat.Dense, r, c int) *mat.Dense {
	if r == 0 || c == 0 {
		return &mat.Dense{}
	}
	if m == nil {
		return mat.NewDense(r, c, nil)
	}
	mr, mc := m.Dims()
	if mr == 0 || mc == 0 {
		return mat.NewDense(r, c, nil)
	}
	return mat.DenseCopyOf(m)
}

func newDense(r, c int) *mat.Dense {
	if r == 0 || c == 0 {
		return &mat.Dense{}
	}
	return mat.NewDense(r, c, nil)
}

func denseNorm(m *mat.Dense) float64 {
	raw := m.RawMatrix()
	sum := 0.0
	for i := 0; i < raw.Rows; i++ {
		row := raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols]
		for _, v := range row {
			sum += v * v
		}
	}
	return math.Sqrt(sum)
}

func eps() float64 {
	return math.Nextafter(1.0, 2.0) - 1.0
}
