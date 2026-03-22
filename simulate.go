package controlsys

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
)

type Response struct {
	Y      *mat.Dense
	XFinal *mat.VecDense
}

type SimulateOpts struct {
	Workspace *mat.VecDense

	yBuf  *mat.Dense
	duBuf *mat.Dense
}

func (sys *System) Simulate(u *mat.Dense, x0 *mat.VecDense, opts *SimulateOpts) (*Response, error) {
	if sys.IsContinuous() {
		return nil, ErrWrongDomain
	}

	if sys.HasInternalDelay() {
		hasIODelay := sys.Delay != nil || sys.InputDelay != nil || sys.OutputDelay != nil
		if hasIODelay {
			merged, err := sys.PullDelaysToLFT()
			if err != nil {
				return nil, err
			}
			return merged.simulateWithInternalDelay(u, x0)
		}
		return sys.simulateWithInternalDelay(u, x0)
	}

	if sys.HasDelay() {
		return sys.simulateWithDelay(u, x0)
	}

	return sys.simulateNoDelay(u, x0, opts)
}

func (sys *System) simulateNoDelay(u *mat.Dense, x0 *mat.VecDense, opts *SimulateOpts) (*Response, error) {
	n, m, p := sys.Dims()

	var steps int
	if u != nil {
		_, steps = u.Dims()
	}

	makeXFinal := func() *mat.VecDense {
		if n == 0 {
			return nil
		}
		v := mat.NewVecDense(n, nil)
		if x0 != nil {
			v.CopyVec(x0)
		}
		return v
	}

	if p == 0 || steps == 0 {
		return &Response{
			Y:      nil,
			XFinal: makeXFinal(),
		}, nil
	}

	var Y *mat.Dense
	if opts != nil && opts.yBuf != nil {
		Y = opts.yBuf
		Y.Zero()
	} else {
		Y = mat.NewDense(p, steps, nil)
	}
	x := makeXFinal()

	if n > 0 {
		var tmp *mat.VecDense
		if opts != nil && opts.Workspace != nil {
			tmp = opts.Workspace
		} else {
			tmp = mat.NewVecDense(n, nil)
		}

		yRaw := Y.RawMatrix()
		aGen := sys.A.RawMatrix()
		cGen := sys.C.RawMatrix()
		xVec := x.RawVector()
		tmpVec := tmp.RawVector()
		ykData := make([]float64, p)
		ykVec := blas64.Vector{N: p, Inc: 1, Data: ykData}

		if m > 0 {
			uColData := make([]float64, m)
			uRaw := u.RawMatrix()
			bGen := sys.B.RawMatrix()
			uColVec := blas64.Vector{N: m, Inc: 1, Data: uColData}

			for k := 0; k < steps; k++ {
				blas64.Gemv(blas.NoTrans, 1, cGen, xVec, 0, ykVec)
				for i := 0; i < p; i++ {
					yRaw.Data[i*yRaw.Stride+k] = ykData[i]
				}

				for j := 0; j < m; j++ {
					uColData[j] = uRaw.Data[j*uRaw.Stride+k]
				}
				blas64.Gemv(blas.NoTrans, 1, aGen, xVec, 0, tmpVec)
				blas64.Gemv(blas.NoTrans, 1, bGen, uColVec, 1, tmpVec)
				x, tmp = tmp, x
				xVec, tmpVec = tmpVec, xVec
			}
		} else {
			for k := 0; k < steps; k++ {
				blas64.Gemv(blas.NoTrans, 1, cGen, xVec, 0, ykVec)
				for i := 0; i < p; i++ {
					yRaw.Data[i*yRaw.Stride+k] = ykData[i]
				}

				blas64.Gemv(blas.NoTrans, 1, aGen, xVec, 0, tmpVec)
				x, tmp = tmp, x
				xVec, tmpVec = tmpVec, xVec
			}
		}
	}

	if m > 0 {
		var du *mat.Dense
		if opts != nil && opts.duBuf != nil {
			du = opts.duBuf
		} else {
			du = mat.NewDense(p, steps, nil)
		}
		du.Mul(sys.D, u)
		Y.Add(Y, du)
	}

	return &Response{
		Y:      Y,
		XFinal: x,
	}, nil
}

func (sys *System) simulateWithDelay(u *mat.Dense, x0 *mat.VecDense) (*Response, error) {
	n, m, p := sys.Dims()

	totalDelay := sys.TotalDelay()
	if totalDelay == nil {
		totalDelay = mat.NewDense(p, m, nil)
	}

	var steps int
	if u != nil {
		_, steps = u.Dims()
	}

	var xFinal *mat.VecDense
	if n > 0 {
		xFinal = mat.NewVecDense(n, nil)
	}

	if p == 0 || steps == 0 {
		if n > 0 && x0 != nil {
			xFinal.CopyVec(x0)
		}
		return &Response{Y: nil, XFinal: xFinal}, nil
	}

	Y := mat.NewDense(p, steps, nil)

	if n > 0 && x0 != nil {
		x := mat.NewVecDense(n, nil)
		x.CopyVec(x0)
		tmp := mat.NewVecDense(n, nil)
		yk := mat.NewVecDense(p, nil)
		yAutoRaw := Y.RawMatrix()
		ykRaw := yk.RawVector()
		for k := 0; k < steps; k++ {
			yk.MulVec(sys.C, x)
			for i := 0; i < p; i++ {
				yAutoRaw.Data[i*yAutoRaw.Stride+k] = ykRaw.Data[i*ykRaw.Inc]
			}
			tmp.MulVec(sys.A, x)
			x, tmp = tmp, x
		}
		xFinal.CopyVec(x)
	}

	if m == 0 {
		return &Response{Y: Y, XFinal: xFinal}, nil
	}

	uRaw := u.RawMatrix()
	dRaw := sys.D.RawMatrix()
	delayRaw := totalDelay.RawMatrix()
	yRaw := Y.RawMatrix()

	delayIdx := make([]int, p*m)
	for i := 0; i < p; i++ {
		for j := 0; j < m; j++ {
			d := int(math.Round(delayRaw.Data[i*delayRaw.Stride+j]))
			if d < 0 {
				return nil, fmt.Errorf("%w: delay(%d,%d) = %d", ErrNegativeDelay, i, j, d)
			}
			delayIdx[i*m+j] = d
		}
	}

	yForced := mat.NewDense(p, m, nil)
	yForcedRaw := yForced.RawMatrix()

	var xCols, nextX *mat.Dense
	var bRaw blas64.General
	if n > 0 {
		xCols = mat.NewDense(n, m, nil)
		nextX = mat.NewDense(n, m, nil)
		bRaw = sys.B.RawMatrix()
	}

	for k := 0; k < steps; k++ {
		var nextRaw blas64.General
		if n > 0 {
			yForced.Mul(sys.C, xCols)
			nextX.Mul(sys.A, xCols)
			nextRaw = nextX.RawMatrix()
		} else {
			yForced.Zero()
		}

		for j := 0; j < m; j++ {
			uk := uRaw.Data[j*uRaw.Stride+k]
			if n > 0 && uk != 0 {
				blas64.Axpy(uk,
					blas64.Vector{N: n, Inc: bRaw.Stride, Data: bRaw.Data[j:]},
					blas64.Vector{N: n, Inc: nextRaw.Stride, Data: nextRaw.Data[j:]},
				)
			}
			for i := 0; i < p; i++ {
				outIdx := k + delayIdx[i*m+j]
				if outIdx >= steps {
					continue
				}
				val := yForcedRaw.Data[i*yForcedRaw.Stride+j] + dRaw.Data[i*dRaw.Stride+j]*uk
				yRaw.Data[i*yRaw.Stride+outIdx] += val
			}
		}

		if n > 0 {
			xCols, nextX = nextX, xCols
		}
	}

	if xFinal != nil {
		xRaw := xCols.RawMatrix()
		xFinalRaw := xFinal.RawVector()
		for i := 0; i < n; i++ {
			sum := xFinalRaw.Data[i*xFinalRaw.Inc]
			for j := 0; j < m; j++ {
				sum += xRaw.Data[i*xRaw.Stride+j]
			}
			xFinalRaw.Data[i*xFinalRaw.Inc] = sum
		}
	}

	return &Response{Y: Y, XFinal: xFinal}, nil
}

func (sys *System) simulateWithInternalDelay(u *mat.Dense, x0 *mat.VecDense) (*Response, error) {
	n, m, p := sys.Dims()
	N := len(sys.InternalDelay)

	var steps int
	if u != nil {
		_, steps = u.Dims()
	}

	makeXFinal := func() *mat.VecDense {
		if n == 0 {
			return nil
		}
		v := mat.NewVecDense(n, nil)
		if x0 != nil {
			v.CopyVec(x0)
		}
		return v
	}

	if p == 0 || steps == 0 {
		return &Response{Y: nil, XFinal: makeXFinal()}, nil
	}

	delays := make([]int, N)
	maxDelay := 0
	for j := 0; j < N; j++ {
		delays[j] = int(math.Round(sys.InternalDelay[j]))
		if delays[j] > maxDelay {
			maxDelay = delays[j]
		}
	}

	bufSize := maxDelay
	if N > 0 && bufSize == 0 {
		bufSize = 1
	}

	Y := mat.NewDense(p, steps, nil)
	x := makeXFinal()

	var zBuf []float64
	if N > 0 {
		zBuf = make([]float64, N*(bufSize+1))
	}

	wData := make([]float64, N)
	zData := make([]float64, N)

	var uColData []float64
	var uData []float64
	var uStride int
	if m > 0 {
		uColData = make([]float64, m)
		if u != nil {
			uRaw := u.RawMatrix()
			uData = uRaw.Data
			uStride = uRaw.Stride
		}
	}

	var tmp *mat.VecDense
	if n > 0 {
		tmp = mat.NewVecDense(n, nil)
	}

	yRaw := Y.RawMatrix()
	ykData := make([]float64, p)

	var aGen, bGen, b2Gen, cGen, dGen, d12Gen, c2Gen, d21Gen, d22Gen blas64.General
	var xVec, tmpVec blas64.Vector
	if n > 0 {
		aGen = sys.A.RawMatrix()
		cGen = sys.C.RawMatrix()
		xVec = x.RawVector()
		tmpVec = tmp.RawVector()
	}
	if m > 0 {
		if n > 0 {
			bGen = sys.B.RawMatrix()
		}
		dGen = sys.D.RawMatrix()
	}
	if N > 0 {
		if n > 0 {
			b2Gen = sys.B2.RawMatrix()
			c2Gen = sys.C2.RawMatrix()
		}
		d12Gen = sys.D12.RawMatrix()
		if m > 0 {
			d21Gen = sys.D21.RawMatrix()
		}
		d22Gen = sys.D22.RawMatrix()
	}
	ykVec := blas64.Vector{N: p, Inc: 1, Data: ykData}
	wVec := blas64.Vector{N: N, Inc: 1, Data: wData}
	zVec := blas64.Vector{N: N, Inc: 1, Data: zData}
	uColVec := blas64.Vector{N: m, Inc: 1, Data: uColData}

	for k := 0; k < steps; k++ {
		for j := 0; j < N; j++ {
			dj := delays[j]
			if dj == 0 {
				if k == 0 {
					wData[j] = 0
				} else {
					wData[j] = zBuf[j*(bufSize+1)+(k-1)%(bufSize+1)]
				}
			} else if k < dj {
				wData[j] = 0
			} else {
				wData[j] = zBuf[j*(bufSize+1)+(k-dj)%(bufSize+1)]
			}
		}

		if m > 0 {
			for i := 0; i < m; i++ {
				uColData[i] = uData[i*uStride+k]
			}
		}

		if n > 0 {
			blas64.Gemv(blas.NoTrans, 1, cGen, xVec, 0, ykVec)
		} else {
			for i := range ykData {
				ykData[i] = 0
			}
		}
		if m > 0 {
			blas64.Gemv(blas.NoTrans, 1, dGen, uColVec, 1, ykVec)
		}
		if N > 0 {
			blas64.Gemv(blas.NoTrans, 1, d12Gen, wVec, 1, ykVec)
		}
		for i := 0; i < p; i++ {
			yRaw.Data[i*yRaw.Stride+k] = ykData[i]
		}

		if N > 0 {
			if n > 0 {
				blas64.Gemv(blas.NoTrans, 1, c2Gen, xVec, 0, zVec)
			} else {
				for i := range zData {
					zData[i] = 0
				}
			}
			if m > 0 {
				blas64.Gemv(blas.NoTrans, 1, d21Gen, uColVec, 1, zVec)
			}
			blas64.Gemv(blas.NoTrans, 1, d22Gen, wVec, 1, zVec)

			idx := k % (bufSize + 1)
			for j := 0; j < N; j++ {
				zBuf[j*(bufSize+1)+idx] = zData[j]
			}
		}

		if n > 0 {
			blas64.Gemv(blas.NoTrans, 1, aGen, xVec, 0, tmpVec)
			if m > 0 {
				blas64.Gemv(blas.NoTrans, 1, bGen, uColVec, 1, tmpVec)
			}
			if N > 0 {
				blas64.Gemv(blas.NoTrans, 1, b2Gen, wVec, 1, tmpVec)
			}
			x, tmp = tmp, x
			xVec, tmpVec = tmpVec, xVec
		}
	}

	return &Response{Y: Y, XFinal: x}, nil
}
