package controlsys

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type ReduceMode int

const (
	ReduceAll            ReduceMode = iota
	ReduceUncontrollable
	ReduceUnobservable
)

type ReduceOpts struct {
	Mode     ReduceMode
	Tol      float64
	Equalize bool
}

type ReduceResult struct {
	Sys        *System
	Order      int
	BlockSizes []int
}

func (sys *System) Reduce(opts *ReduceOpts) (*ReduceResult, error) {
	if opts == nil {
		opts = &ReduceOpts{}
	}

	n, m, p := sys.Dims()

	if n == 0 {
		return &ReduceResult{Sys: copySys(sys, n, m, p), Order: 0}, nil
	}

	if m == 0 && (opts.Mode == ReduceAll || opts.Mode == ReduceUncontrollable) {
		return zeroOrderResult(sys, m, p), nil
	}
	if p == 0 && (opts.Mode == ReduceAll || opts.Mode == ReduceUnobservable) {
		return zeroOrderResult(sys, m, p), nil
	}

	A := mat.DenseCopyOf(sys.A)
	B := mat.DenseCopyOf(sys.B)
	C := mat.DenseCopyOf(sys.C)

	if opts.Equalize {
		equalize(A, B, C, n, m, p)
	}

	ncont := n
	var blockSizes []int

	if opts.Mode == ReduceAll || opts.Mode == ReduceUncontrollable {
		res := ControllabilityStaircase(A, B, C, opts.Tol)
		A = res.A
		B = res.B
		C = res.C
		ncont = res.NCont
		blockSizes = res.BlockSizes
	}

	nr := ncont

	if opts.Mode == ReduceAll || opts.Mode == ReduceUnobservable {
		if ncont == 0 {
			nr = 0
		} else {
			ac := extractSubmatrix(A, 0, ncont, 0, ncont)
			bc := extractSubmatrix(B, 0, ncont, 0, m)
			cc := extractSubmatrix(C, 0, p, 0, ncont)

			acT := mat.DenseCopyOf(ac.T())
			ccT := mat.DenseCopyOf(cc.T())
			bcT := mat.DenseCopyOf(bc.T())

			dualRes := ControllabilityStaircase(acT, ccT, bcT, opts.Tol)
			nobs := dualRes.NCont

			if nobs == 0 {
				nr = 0
			} else {
				nr = nobs

				// Undo duality: transpose the observable block
				aObsDual := extractSubmatrix(dualRes.A, 0, nobs, 0, nobs)
				aObs := mat.DenseCopyOf(aObsDual.T())

				// B_min = dualRes.C[:,:nobs]' (nobs×m)
				bObs := mat.DenseCopyOf(extractSubmatrix(dualRes.C, 0, m, 0, nobs).T())

				// C_min = dualRes.B[:nobs,:]' (p×nobs)
				cObs := mat.DenseCopyOf(extractSubmatrix(dualRes.B, 0, nobs, 0, p).T())

				// Pertranspose for upper block Hessenberg: P*A*P, P*B, C*P
				pertransposeSquare(aObs, nobs)
				reverseRows(bObs, nobs)
				reverseCols(cObs, nobs)

				A = aObs
				B = bObs
				C = cObs
			}
		}
	}

	ar := extractSubmatrix(A, 0, nr, 0, nr)
	br := extractSubmatrix(B, 0, nr, 0, m)
	cr := extractSubmatrix(C, 0, p, 0, nr)
	dr := denseCopySafe(sys.D, p, m)

	reduced := &System{A: ar, B: br, C: cr, D: dr, Delay: copyDelayOrNil(sys.Delay), Dt: sys.Dt}
	propagateIONames(reduced, sys)

	return &ReduceResult{
		Sys:        reduced,
		Order:      nr,
		BlockSizes: blockSizes,
	}, nil
}

func (sys *System) MinimalRealization() (*ReduceResult, error) {
	return sys.Reduce(nil)
}

func copySys(sys *System, n, m, p int) *System {
	cp := &System{
		A:     denseCopySafe(sys.A, n, n),
		B:     denseCopySafe(sys.B, n, m),
		C:     denseCopySafe(sys.C, p, n),
		D:     denseCopySafe(sys.D, p, m),
		Delay: copyDelayOrNil(sys.Delay),
		Dt:    sys.Dt,
	}
	propagateIONames(cp, sys)
	return cp
}


func zeroOrderResult(sys *System, m, p int) *ReduceResult {
	s := &System{
		A:     &mat.Dense{},
		B:     &mat.Dense{},
		C:     &mat.Dense{},
		D:     denseCopySafe(sys.D, p, m),
		Delay: copyDelayOrNil(sys.Delay),
		Dt:    sys.Dt,
	}
	propagateIONames(s, sys)
	return &ReduceResult{
		Sys:   s,
		Order: 0,
	}
}

func extractSubmatrix(m *mat.Dense, r0, r1, c0, c1 int) *mat.Dense {
	rows := r1 - r0
	cols := c1 - c0
	if rows <= 0 || cols <= 0 {
		return &mat.Dense{}
	}
	raw := m.RawMatrix()
	data := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		copy(data[i*cols:], raw.Data[(r0+i)*raw.Stride+c0:(r0+i)*raw.Stride+c0+cols])
	}
	return mat.NewDense(rows, cols, data)
}

func pertransposeSquare(a *mat.Dense, n int) {
	if n <= 1 {
		return
	}
	raw := a.RawMatrix()
	for i := 0; i < n/2; i++ {
		j := n - 1 - i
		ri := raw.Data[i*raw.Stride:]
		rj := raw.Data[j*raw.Stride:]
		for k := 0; k < n; k++ {
			ri[k], rj[k] = rj[k], ri[k]
		}
	}
	for k := 0; k < n; k++ {
		rk := raw.Data[k*raw.Stride:]
		for i := 0; i < n/2; i++ {
			j := n - 1 - i
			rk[i], rk[j] = rk[j], rk[i]
		}
	}
}

func reverseRows(m *mat.Dense, n int) {
	if n <= 1 {
		return
	}
	raw := m.RawMatrix()
	cols := raw.Cols
	for i := 0; i < n/2; i++ {
		j := n - 1 - i
		ri := raw.Data[i*raw.Stride : i*raw.Stride+cols]
		rj := raw.Data[j*raw.Stride : j*raw.Stride+cols]
		for k := range ri {
			ri[k], rj[k] = rj[k], ri[k]
		}
	}
}

func reverseCols(m *mat.Dense, n int) {
	if n <= 1 {
		return
	}
	raw := m.RawMatrix()
	for k := 0; k < raw.Rows; k++ {
		rk := raw.Data[k*raw.Stride:]
		for i := 0; i < n/2; i++ {
			j := n - 1 - i
			rk[i], rk[j] = rk[j], rk[i]
		}
	}
}

func equalize(A, B, C *mat.Dense, n, m, p int) {
	const sclfac = 10.0
	const factor = 0.95
	sfmin1 := math.SmallestNonzeroFloat64 / eps()
	sfmax1 := 1.0 / sfmin1

	scale := make([]float64, n)
	for i := range scale {
		scale[i] = 1.0
	}

	aRaw := A.RawMatrix()
	bRaw := B.RawMatrix()
	cRaw := C.RawMatrix()

	for {
		noconv := false
		for i := 0; i < n; i++ {
			c := 0.0
			r := 0.0

			aRow := aRaw.Data[i*aRaw.Stride:]
			for j := 0; j < n; j++ {
				if j == i {
					continue
				}
				c += math.Abs(aRaw.Data[j*aRaw.Stride+i])
				r += math.Abs(aRow[j])
			}
			bRow := bRaw.Data[i*bRaw.Stride:]
			for j := 0; j < m; j++ {
				r += math.Abs(bRow[j])
			}
			for j := 0; j < p; j++ {
				c += math.Abs(cRaw.Data[j*cRaw.Stride+i])
			}

			if c == 0 || r == 0 {
				continue
			}

			g := r / sclfac
			f := 1.0
			s := c + r

			for c < g {
				if f*scale[i] > sfmax1/sclfac {
					break
				}
				f *= sclfac
				c *= sclfac * sclfac
			}

			g = r * sclfac
			for c > g {
				if scale[i]*f <= sfmin1*sclfac {
					break
				}
				f /= sclfac
				c /= sclfac * sclfac
			}

			if (c+r)/f >= factor*s {
				continue
			}

			scale[i] *= f
			noconv = true

			for j := 0; j < n; j++ {
				aRow[j] *= f
			}
			for j := 0; j < m; j++ {
				bRow[j] *= f
			}
			fi := 1.0 / f
			for j := 0; j < n; j++ {
				aRaw.Data[j*aRaw.Stride+i] *= fi
			}
			for j := 0; j < p; j++ {
				cRaw.Data[j*cRaw.Stride+i] *= fi
			}
		}
		if !noconv {
			break
		}
	}
}
