package controlsys

import (
	"math"

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
	buk   *mat.VecDense
	yk    *mat.VecDense
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

		var buk *mat.VecDense
		if opts != nil && opts.buk != nil {
			buk = opts.buk
		} else {
			buk = mat.NewVecDense(n, nil)
		}
		var yk *mat.VecDense
		if opts != nil && opts.yk != nil {
			yk = opts.yk
		} else {
			yk = mat.NewVecDense(p, nil)
		}

		yRaw := Y.RawMatrix()
		ykRaw := yk.RawVector()

		if m > 0 {
			uColData := make([]float64, m)
			uCol := mat.NewVecDense(m, uColData)
			uRaw := u.RawMatrix()

			for k := 0; k < steps; k++ {
				yk.MulVec(sys.C, x)
				for i := 0; i < p; i++ {
					yRaw.Data[i*yRaw.Stride+k] = ykRaw.Data[i*ykRaw.Inc]
				}

				tmp.MulVec(sys.A, x)
				for j := 0; j < m; j++ {
					uColData[j] = uRaw.Data[j*uRaw.Stride+k]
				}
				buk.MulVec(sys.B, uCol)
				tmp.AddVec(tmp, buk)
				x, tmp = tmp, x
			}
		} else {
			for k := 0; k < steps; k++ {
				yk.MulVec(sys.C, x)
				for i := 0; i < p; i++ {
					yRaw.Data[i*yRaw.Stride+k] = ykRaw.Data[i*ykRaw.Inc]
				}

				tmp.MulVec(sys.A, x)
				x, tmp = tmp, x
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
			delayIdx[i*m+j] = int(math.Round(delayRaw.Data[i*delayRaw.Stride+j]))
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
				for i := 0; i < n; i++ {
					nextRaw.Data[i*nextRaw.Stride+j] += bRaw.Data[i*bRaw.Stride+j] * uk
				}
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
			row := xRaw.Data[i*xRaw.Stride : i*xRaw.Stride+m]
			for _, v := range row {
				sum += v
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

	w := mat.NewVecDense(N, nil)
	z := mat.NewVecDense(N, nil)

	var uCol *mat.VecDense
	var uColData []float64
	var uData []float64
	var uStride int
	if m > 0 {
		uColData = make([]float64, m)
		uCol = mat.NewVecDense(m, uColData)
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
	wRaw := w.RawVector()
	zRaw := z.RawVector()
	yk := mat.NewVecDense(p, nil)
	ykBuf := mat.NewVecDense(p, nil)
	zBuf2 := mat.NewVecDense(N, nil)
	var buk, b2w *mat.VecDense
	if n > 0 {
		b2w = mat.NewVecDense(n, nil)
		if m > 0 {
			buk = mat.NewVecDense(n, nil)
		}
	}

	for k := 0; k < steps; k++ {
		for j := 0; j < N; j++ {
			dj := delays[j]
			if dj == 0 {
				if k == 0 {
					wRaw.Data[j*wRaw.Inc] = 0
				} else {
					idx := (k - 1) % (bufSize + 1)
					wRaw.Data[j*wRaw.Inc] = zBuf[j*(bufSize+1)+idx]
				}
			} else if k < dj {
				wRaw.Data[j*wRaw.Inc] = 0
			} else {
				idx := (k - dj) % (bufSize + 1)
				wRaw.Data[j*wRaw.Inc] = zBuf[j*(bufSize+1)+idx]
			}
		}

		if m > 0 {
			for i := 0; i < m; i++ {
				uColData[i] = uData[i*uStride+k]
			}
		}

		yk.Zero()
		if n > 0 {
			yk.MulVec(sys.C, x)
		}
		if m > 0 {
			ykBuf.MulVec(sys.D, uCol)
			yk.AddVec(yk, ykBuf)
		}
		if N > 0 {
			ykBuf.MulVec(sys.D12, w)
			yk.AddVec(yk, ykBuf)
		}
		ykRaw := yk.RawVector()
		for i := 0; i < p; i++ {
			yRaw.Data[i*yRaw.Stride+k] = ykRaw.Data[i*ykRaw.Inc]
		}

		z.Zero()
		if N > 0 {
			if n > 0 {
				z.MulVec(sys.C2, x)
			}
			if m > 0 {
				zBuf2.MulVec(sys.D21, uCol)
				z.AddVec(z, zBuf2)
			}
			zBuf2.MulVec(sys.D22, w)
			z.AddVec(z, zBuf2)

			idx := k % (bufSize + 1)
			for j := 0; j < N; j++ {
				zBuf[j*(bufSize+1)+idx] = zRaw.Data[j*zRaw.Inc]
			}
		}

		if n > 0 {
			tmp.MulVec(sys.A, x)
			if m > 0 {
				buk.MulVec(sys.B, uCol)
				tmp.AddVec(tmp, buk)
			}
			if N > 0 {
				b2w.MulVec(sys.B2, w)
				tmp.AddVec(tmp, b2w)
			}
			x, tmp = tmp, x
		}
	}

	return &Response{Y: Y, XFinal: x}, nil
}
