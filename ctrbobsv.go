package controlsys

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
)

// Ctrb returns the n×(n·m) controllability matrix [B, AB, A²B, …, Aⁿ⁻¹B].
func Ctrb(A, B *mat.Dense) (*mat.Dense, error) {
	n, nc := A.Dims()
	if n != nc {
		return nil, ErrDimensionMismatch
	}
	br, m := B.Dims()
	if br != n {
		return nil, ErrDimensionMismatch
	}
	if n == 0 {
		return &mat.Dense{}, nil
	}

	cols := n * m
	data := make([]float64, n*cols)
	bRaw := B.RawMatrix()
	for i := range n {
		copy(data[i*cols:i*cols+m], bRaw.Data[i*bRaw.Stride:i*bRaw.Stride+m])
	}

	aGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: make([]float64, n*n)}
	aRaw := A.RawMatrix()
	for i := range n {
		copy(aGen.Data[i*n:i*n+n], aRaw.Data[i*aRaw.Stride:i*aRaw.Stride+n])
	}

	for k := 1; k < n; k++ {
		prev := blas64.General{Rows: n, Cols: m, Stride: cols, Data: data[(k-1)*m:]}
		cur := blas64.General{Rows: n, Cols: m, Stride: cols, Data: data[k*m:]}
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, aGen, prev, 0, cur)
	}

	return mat.NewDense(n, cols, data), nil
}

// Obsv returns the (n·p)×n observability matrix [C; CA; CA²; …; CAⁿ⁻¹].
func Obsv(A, C *mat.Dense) (*mat.Dense, error) {
	n, nc := A.Dims()
	if n != nc {
		return nil, ErrDimensionMismatch
	}
	p, cc := C.Dims()
	if cc != n {
		return nil, ErrDimensionMismatch
	}
	if n == 0 {
		return &mat.Dense{}, nil
	}

	rows := n * p
	data := make([]float64, rows*n)
	cRaw := C.RawMatrix()
	for i := range p {
		copy(data[i*n:i*n+n], cRaw.Data[i*cRaw.Stride:i*cRaw.Stride+n])
	}

	aGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: make([]float64, n*n)}
	aRaw := A.RawMatrix()
	for i := range n {
		copy(aGen.Data[i*n:i*n+n], aRaw.Data[i*aRaw.Stride:i*aRaw.Stride+n])
	}

	for k := 1; k < n; k++ {
		prev := blas64.General{Rows: p, Cols: n, Stride: n, Data: data[(k-1)*p*n:]}
		cur := blas64.General{Rows: p, Cols: n, Stride: n, Data: data[k*p*n:]}
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, prev, aGen, 0, cur)
	}

	return mat.NewDense(rows, n, data), nil
}

// CtrbF computes the controllability staircase form.
// C may be nil if not needed in the transformed system.
func CtrbF(A, B, C *mat.Dense) (*StaircaseResult, error) {
	n, nc := A.Dims()
	if n != nc {
		return nil, ErrDimensionMismatch
	}
	br, _ := B.Dims()
	if br != n {
		return nil, ErrDimensionMismatch
	}
	if C != nil {
		_, cc := C.Dims()
		if cc != n {
			return nil, ErrDimensionMismatch
		}
	}
	return ControllabilityStaircase(A, B, C, 0), nil
}

// ObsvF computes the observability staircase form via duality.
// B may be nil if not needed in the transformed system.
func ObsvF(A, B, C *mat.Dense) (*StaircaseResult, error) {
	n, nc := A.Dims()
	if n != nc {
		return nil, ErrDimensionMismatch
	}
	p, cc := C.Dims()
	if cc != n {
		return nil, ErrDimensionMismatch
	}
	if B != nil {
		br, _ := B.Dims()
		if br != n {
			return nil, ErrDimensionMismatch
		}
	}

	acT := mat.DenseCopyOf(A.T())
	ccT := mat.DenseCopyOf(C.T())
	var bcT *mat.Dense
	if B != nil {
		bcT = mat.DenseCopyOf(B.T())
	}

	dual := ControllabilityStaircase(acT, ccT, bcT, 0)

	nobs := dual.NCont
	res := &StaircaseResult{
		NCont:      nobs,
		BlockSizes: dual.BlockSizes,
	}

	res.A = mat.DenseCopyOf(dual.A.T())

	if dual.C != nil {
		_, mc := dual.C.Dims()
		if mc > 0 {
			res.B = mat.DenseCopyOf(dual.C.T())
		} else {
			res.B = &mat.Dense{}
		}
	} else {
		res.B = &mat.Dense{}
	}

	dr, _ := dual.B.Dims()
	if dr > 0 {
		res.C = mat.DenseCopyOf(dual.B.T())
	} else {
		res.C = &mat.Dense{}
	}

	_ = p
	return res, nil
}
