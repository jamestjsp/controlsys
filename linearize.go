package controlsys

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

type NonlinearModel struct {
	F func(x, u *mat.VecDense) *mat.VecDense
	H func(x, u *mat.VecDense) *mat.VecDense
	N int
	M int
	P int
}

func Linearize(model *NonlinearModel, x0, u0 *mat.VecDense) (*System, error) {
	if model == nil {
		return nil, fmt.Errorf("controlsys: nil model: %w", ErrDimensionMismatch)
	}
	if model.F == nil || model.H == nil {
		return nil, fmt.Errorf("controlsys: nil F or H function: %w", ErrDimensionMismatch)
	}
	if x0 == nil || u0 == nil {
		return nil, fmt.Errorf("controlsys: nil x0 or u0: %w", ErrDimensionMismatch)
	}
	if x0.Len() != model.N {
		return nil, fmt.Errorf("controlsys: x0 length %d != N %d: %w", x0.Len(), model.N, ErrDimensionMismatch)
	}
	if u0.Len() != model.M {
		return nil, fmt.Errorf("controlsys: u0 length %d != M %d: %w", u0.Len(), model.M, ErrDimensionMismatch)
	}

	sqrtEps := math.Sqrt(eps())
	n, m, p := model.N, model.M, model.P

	aData := make([]float64, n*n)
	bData := make([]float64, n*m)
	cData := make([]float64, p*n)
	dData := make([]float64, p*m)

	xp := mat.NewVecDense(n, nil)
	xm := mat.NewVecDense(n, nil)

	for j := 0; j < n; j++ {
		h := sqrtEps * math.Max(math.Abs(x0.AtVec(j)), 1.0)
		inv2h := 1.0 / (2.0 * h)

		xp.CopyVec(x0)
		xm.CopyVec(x0)
		xp.SetVec(j, x0.AtVec(j)+h)
		xm.SetVec(j, x0.AtVec(j)-h)

		fp := model.F(xp, u0)
		fm := model.F(xm, u0)
		for i := 0; i < n; i++ {
			aData[i*n+j] = (fp.AtVec(i) - fm.AtVec(i)) * inv2h
		}

		hp := model.H(xp, u0)
		hm := model.H(xm, u0)
		for i := 0; i < p; i++ {
			cData[i*n+j] = (hp.AtVec(i) - hm.AtVec(i)) * inv2h
		}
	}

	up := mat.NewVecDense(m, nil)
	um := mat.NewVecDense(m, nil)

	for j := 0; j < m; j++ {
		h := sqrtEps * math.Max(math.Abs(u0.AtVec(j)), 1.0)
		inv2h := 1.0 / (2.0 * h)

		up.CopyVec(u0)
		um.CopyVec(u0)
		up.SetVec(j, u0.AtVec(j)+h)
		um.SetVec(j, u0.AtVec(j)-h)

		fp := model.F(x0, up)
		fm := model.F(x0, um)
		for i := 0; i < n; i++ {
			bData[i*m+j] = (fp.AtVec(i) - fm.AtVec(i)) * inv2h
		}

		hp := model.H(x0, up)
		hm := model.H(x0, um)
		for i := 0; i < p; i++ {
			dData[i*m+j] = (hp.AtVec(i) - hm.AtVec(i)) * inv2h
		}
	}

	A := mat.NewDense(n, n, aData)
	B := mat.NewDense(n, m, bData)
	C := mat.NewDense(p, n, cData)
	D := mat.NewDense(p, m, dData)

	return New(A, B, C, D, 0)
}
