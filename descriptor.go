package controlsys

import (
	"fmt"

	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/gonum/mat"
)

func copyDescriptorE(E *mat.Dense) *mat.Dense {
	if E == nil {
		return nil
	}
	return mat.DenseCopyOf(E)
}

// generalizedPoles computes the finite eigenvalues of the pencil (A, E)
// via the QZ algorithm (DGGES). Infinite eigenvalues (beta=0) are excluded.
func generalizedPoles(A, E *mat.Dense, n int) ([]complex128, error) {
	aData := make([]float64, n*n)
	eData := make([]float64, n*n)
	aRaw := A.RawMatrix()
	eRaw := E.RawMatrix()
	copyStrided(aData, n, aRaw.Data, aRaw.Stride, n, n)
	copyStrided(eData, n, eRaw.Data, eRaw.Stride, n, n)

	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	vsl := make([]float64, n*n)
	vsr := make([]float64, n*n)

	var workQuery [1]float64
	impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
		n, aData, n, eData, n, alphar, alphai, beta, vsl, n, vsr, n, workQuery[:], -1, nil)
	lwork := int(workQuery[0])
	work := make([]float64, lwork)
	_, ok := impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
		n, aData, n, eData, n, alphar, alphai, beta, vsl, n, vsr, n, work, lwork, nil)
	if !ok {
		return nil, fmt.Errorf("controlsys: generalized eigenvalue decomposition failed")
	}

	poles := make([]complex128, 0, n)
	for i := range n {
		if beta[i] == 0 {
			continue
		}
		poles = append(poles, complex(alphar[i]/beta[i], alphai[i]/beta[i]))
	}
	return poles, nil
}
