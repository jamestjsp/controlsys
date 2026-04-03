package controlsys

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func Covar(sys *System, W *mat.Dense) (*mat.Dense, error) {
	n, m, p := sys.Dims()

	wr, wc := W.Dims()
	if wr != m || wc != m {
		return nil, fmt.Errorf("W must be %d×%d, got %d×%d: %w", m, m, wr, wc, ErrDimensionMismatch)
	}

	stable, err := sys.IsStable()
	if err != nil {
		return nil, err
	}
	if !stable {
		return nil, ErrUnstable
	}

	// Q = B*W*B'
	var BW mat.Dense
	BW.Mul(sys.B, W)
	Q := mat.NewDense(n, n, nil)
	Q.Mul(&BW, sys.B.T())

	// symmetrize Q for Lyapunov solver
	qRaw := Q.RawMatrix()
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			avg := (qRaw.Data[i*qRaw.Stride+j] + qRaw.Data[j*qRaw.Stride+i]) / 2
			qRaw.Data[i*qRaw.Stride+j] = avg
			qRaw.Data[j*qRaw.Stride+i] = avg
		}
	}

	var X *mat.Dense
	if sys.IsContinuous() {
		X, err = Lyap(sys.A, Q, nil)
	} else {
		X, err = DLyap(sys.A, Q, nil)
	}
	if err != nil {
		return nil, err
	}

	// P = C*X*C' + D*W*D'
	var CX mat.Dense
	CX.Mul(sys.C, X)
	P := mat.NewDense(p, p, nil)
	P.Mul(&CX, sys.C.T())

	var DW mat.Dense
	DW.Mul(sys.D, W)
	var DWDt mat.Dense
	DWDt.Mul(&DW, sys.D.T())
	P.Add(P, &DWDt)

	return P, nil
}
