package controlsys

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type SsbalResult struct {
	Sys *System
	T   *mat.Dense
}

func Ssbal(sys *System) (*SsbalResult, error) {
	n, m, p := sys.Dims()
	if n == 0 {
		cp := sys.Copy()
		eye := &mat.Dense{}
		return &SsbalResult{Sys: cp, T: eye}, nil
	}

	aRaw := sys.A.RawMatrix()
	bRaw := sys.B.RawMatrix()
	cRaw := sys.C.RawMatrix()

	d := make([]float64, n)
	for i := range d {
		d[i] = 1.0
	}

	// Iterative diagonal balancing
	const maxIter = 20
	const beta = 2.0
	for iter := 0; iter < maxIter; iter++ {
		changed := false
		for i := 0; i < n; i++ {
			rowNorm := 0.0
			colNorm := 0.0

			for j := 0; j < n; j++ {
				rowNorm += math.Abs(aRaw.Data[i*aRaw.Stride+j]) * d[j]
				colNorm += math.Abs(aRaw.Data[j*aRaw.Stride+i]) * d[j]
			}

			for j := 0; j < m; j++ {
				rowNorm += math.Abs(bRaw.Data[i*bRaw.Stride+j])
			}

			for j := 0; j < p; j++ {
				colNorm += math.Abs(cRaw.Data[j*cRaw.Stride+i])
			}

			if rowNorm == 0 || colNorm == 0 {
				continue
			}

			f := 1.0
			s := rowNorm + colNorm
			// Scale by powers of beta to find best balance
			for rowNorm < colNorm/beta {
				rowNorm *= beta
				colNorm /= beta
				f *= beta
			}
			for rowNorm > colNorm*beta {
				rowNorm /= beta
				colNorm *= beta
				f /= beta
			}

			if rowNorm+colNorm < 0.95*s {
				changed = true
				d[i] *= f
			}
		}
		if !changed {
			break
		}
	}

	Anew := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			Anew.Set(i, j, (d[i]/d[j])*aRaw.Data[i*aRaw.Stride+j])
		}
	}

	Bnew := mat.NewDense(n, m, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			Bnew.Set(i, j, d[i]*bRaw.Data[i*bRaw.Stride+j])
		}
	}

	Cnew := mat.NewDense(p, n, nil)
	for i := 0; i < p; i++ {
		for j := 0; j < n; j++ {
			Cnew.Set(i, j, cRaw.Data[i*cRaw.Stride+j]/d[j])
		}
	}

	Dnew := denseCopy(sys.D)

	T := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		T.Set(i, i, d[i])
	}

	newSys, err := newNoCopy(Anew, Bnew, Cnew, Dnew, sys.Dt)
	if err != nil {
		return nil, err
	}
	propagateIONames(newSys, sys)

	return &SsbalResult{Sys: newSys, T: T}, nil
}
