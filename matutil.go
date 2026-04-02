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

func isSymmetric(m *mat.Dense, tol float64) bool {
	r, c := m.Dims()
	if r != c {
		return false
	}
	raw := m.RawMatrix()
	for i := 0; i < r; i++ {
		for j := i + 1; j < c; j++ {
			if math.Abs(raw.Data[i*raw.Stride+j]-raw.Data[j*raw.Stride+i]) > tol {
				return false
			}
		}
	}
	return true
}

func symmetrize(data []float64, n, stride int) {
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			avg := 0.5 * (data[i*stride+j] + data[j*stride+i])
			data[i*stride+j] = avg
			data[j*stride+i] = avg
		}
	}
}

func copyStrided(dst []float64, dstStride int, src []float64, srcStride int, rows, cols int) {
	for i := 0; i < rows; i++ {
		copy(dst[i*dstStride:i*dstStride+cols], src[i*srcStride:i*srcStride+cols])
	}
}

func copyBlock(dst []float64, dstStride, dstR0, dstC0 int, src []float64, srcStride, srcR0, srcC0 int, rows, cols int) {
	copyStrided(dst[dstR0*dstStride+dstC0:], dstStride, src[srcR0*srcStride+srcC0:], srcStride, rows, cols)
}

func transposeDenseInto(dst []float64, src *mat.Dense) *mat.Dense {
	r, c := src.Dims()
	raw := src.RawMatrix()
	for i := range r {
		for j := range c {
			dst[j*r+i] = raw.Data[i*raw.Stride+j]
		}
	}
	return mat.NewDense(c, r, dst)
}

func extractBlock(m *mat.Dense, r0, c0, rows, cols int) *mat.Dense {
	if rows == 0 || cols == 0 {
		return mat.NewDense(max(rows, 1), max(cols, 1), nil)
	}
	raw := m.RawMatrix()
	data := make([]float64, rows*cols)
	copyBlock(data, cols, 0, 0, raw.Data, raw.Stride, r0, c0, rows, cols)
	return mat.NewDense(rows, cols, data)
}
