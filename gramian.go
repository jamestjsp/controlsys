package controlsys

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
)

type GramType int

const (
	GramControllability GramType = iota
	GramObservability
)

type GramResult struct {
	X *mat.Dense
	L *mat.Dense
}

// Gram computes the controllability or observability gramian of a stable LTI system.
//
// Controllability gramian Wc satisfies:
//
//	Continuous: A·Wc + Wc·A' + B·B' = 0
//	Discrete:   A·Wc·A' - Wc + B·B' = 0
//
// Observability gramian Wo satisfies:
//
//	Continuous: A'·Wo + Wo·A + C'·C = 0
//	Discrete:   A'·Wo·A - Wo + C'·C = 0
func Gram(sys *System, typ GramType) (*GramResult, error) {
	n, m, p := sys.Dims()
	if n == 0 {
		return &GramResult{X: &mat.Dense{}}, nil
	}

	stable, err := sys.IsStable()
	if err != nil {
		return nil, err
	}
	if !stable {
		return nil, ErrUnstableGramian
	}

	aRaw := sys.A.RawMatrix()
	aData := make([]float64, n*n)
	copyStrided(aData, n, aRaw.Data, aRaw.Stride, n, n)

	var q []float64
	var solveA []float64

	switch typ {
	case GramControllability:
		q = make([]float64, n*n)
		bRaw := sys.B.RawMatrix()
		bGen := blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data}
		qGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: q}
		blas64.Gemm(blas.NoTrans, blas.Trans, 1, bGen, bGen, 0, qGen)
		symmetrize(q, n, n)
		solveA = aData

	case GramObservability:
		q = make([]float64, n*n)
		cRaw := sys.C.RawMatrix()
		cGen := blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data}
		qGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: q}
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, cGen, cGen, 0, qGen)
		symmetrize(q, n, n)
		solveA = make([]float64, n*n)
		for i := range n {
			for j := range n {
				solveA[i*n+j] = aData[j*n+i]
			}
		}

	default:
		return nil, ErrDimensionMismatch
	}

	Q := mat.NewDense(n, n, q)
	Aarg := mat.NewDense(n, n, solveA)

	var X *mat.Dense
	if sys.IsContinuous() {
		X, err = Lyap(Aarg, Q)
	} else {
		X, err = DLyap(Aarg, Q)
	}
	if err != nil {
		return nil, err
	}

	res := &GramResult{X: X}

	xRaw := X.RawMatrix()
	lData := make([]float64, n*n)
	copyStrided(lData, n, xRaw.Data, xRaw.Stride, n, n)
	ok := impl.Dpotrf(blas.Upper, n, lData, n)
	if ok {
		res.L = mat.NewDense(n, n, lData)
		for i := range n {
			for j := range i {
				lData[i*n+j] = 0
			}
		}
	}

	return res, nil
}
