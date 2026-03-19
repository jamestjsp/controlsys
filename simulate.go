package controlsys

import (
	"math"

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
	ws := mat.NewVecDense(max(n, 1), nil)
	var bukBuf *mat.VecDense
	if n > 0 {
		bukBuf = mat.NewVecDense(n, nil)
	}
	ykBuf := mat.NewVecDense(p, nil)
	simoY := mat.NewDense(p, steps, nil)
	duBuf := mat.NewDense(p, steps, nil)
	opts := &SimulateOpts{
		Workspace: ws,
		yBuf:      simoY,
		buk:       bukBuf,
		yk:        ykBuf,
		duBuf:     duBuf,
	}

	if n > 0 && x0 != nil {
		x := mat.NewVecDense(n, nil)
		x.CopyVec(x0)
		tmp := ws
		yk := ykBuf
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

	var bCol *mat.Dense
	var bColRaw []float64
	if n > 0 {
		bCol = mat.NewDense(n, 1, nil)
		bColRaw = bCol.RawMatrix().Data
	} else {
		bCol = &mat.Dense{}
	}
	dCol := mat.NewDense(p, 1, nil)
	dColRaw := dCol.RawMatrix().Data
	uj := mat.NewDense(1, steps, nil)
	ujRaw := uj.RawMatrix().Data

	simoSys := &System{A: sys.A, B: bCol, C: sys.C, D: dCol, Dt: sys.Dt}
	uRaw := u.RawMatrix()
	bRaw := sys.B.RawMatrix()
	dRaw := sys.D.RawMatrix()
	delayRaw := totalDelay.RawMatrix()
	yRaw := Y.RawMatrix()

	for j := 0; j < m; j++ {
		if n > 0 {
			for i := 0; i < n; i++ {
				bColRaw[i] = bRaw.Data[i*bRaw.Stride+j]
			}
		}
		for i := 0; i < p; i++ {
			dColRaw[i] = dRaw.Data[i*dRaw.Stride+j]
		}
		copy(ujRaw, uRaw.Data[j*uRaw.Stride:j*uRaw.Stride+steps])

		simoResp, err := simoSys.simulateNoDelay(uj, nil, opts)
		if err != nil {
			return nil, err
		}

		sRaw := simoResp.Y.RawMatrix()
		for i := 0; i < p; i++ {
			d := int(math.Round(delayRaw.Data[i*delayRaw.Stride+j]))
			yRow := yRaw.Data[i*yRaw.Stride:]
			sRow := sRaw.Data[i*sRaw.Stride:]
			for k := d; k < steps; k++ {
				yRow[k] += sRow[k-d]
			}
		}

		if xFinal != nil {
			xFinal.AddVec(xFinal, simoResp.XFinal)
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
