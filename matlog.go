package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/mat"
)

// matLog computes the principal matrix logarithm of a real square matrix via
// eigendecomposition: A = V·Λ·V⁻¹ ⇒ log(A) = V·log(Λ)·V⁻¹.
//
// Returns ErrSingularTransform if A has an eigenvalue on the branch cut
// (real, ≤ 0) or if the eigenvector matrix is singular/ill-conditioned
// (defective / near-defective A).
func matLog(A *mat.Dense) (*mat.Dense, error) {
	n, c := A.Dims()
	if n != c {
		return nil, fmt.Errorf("matLog: non-square %dx%d", n, c)
	}
	if n == 0 {
		return mat.NewDense(0, 0, nil), nil
	}

	var eig mat.Eigen
	if !eig.Factorize(A, mat.EigenRight) {
		return nil, fmt.Errorf("matLog: eigendecomposition failed: %w", ErrSchurFailed)
	}
	vals := eig.Values(nil)

	for i, v := range vals {
		if imag(v) == 0 && real(v) <= 0 {
			return nil, fmt.Errorf("matLog: eigenvalue[%d]=%v on non-positive real axis (branch cut): %w",
				i, v, ErrSingularTransform)
		}
		if cmplx.Abs(v) == 0 {
			return nil, fmt.Errorf("matLog: eigenvalue[%d]=0: %w", i, ErrSingularTransform)
		}
	}

	var vecC mat.CDense
	eig.VectorsTo(&vecC)

	V := mat.NewCDense(n, n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			V.Set(i, j, vecC.At(i, j))
		}
	}

	Vinv, err := cinvert(V, n)
	if err != nil {
		return nil, fmt.Errorf("matLog: eigenvector matrix singular (defective A): %w", ErrSingularTransform)
	}

	cond := cmatNorm(V, n) * cmatNorm(Vinv, n)
	if cond > 1e12 || math.IsNaN(cond) || math.IsInf(cond, 0) {
		return nil, fmt.Errorf("matLog: eigenvector matrix ill-conditioned (cond=%.2e), A may be defective: %w",
			cond, ErrSingularTransform)
	}

	DVinv := mat.NewCDense(n, n, nil)
	for i := 0; i < n; i++ {
		lv := cmplx.Log(vals[i])
		for j := 0; j < n; j++ {
			DVinv.Set(i, j, lv*Vinv.At(i, j))
		}
	}

	L := mat.NewCDense(n, n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var s complex128
			for k := 0; k < n; k++ {
				s += V.At(i, k) * DVinv.At(k, j)
			}
			L.Set(i, j, s)
		}
	}

	result := mat.NewDense(n, n, nil)
	maxIm, maxRe := 0.0, 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			re := real(L.At(i, j))
			im := math.Abs(imag(L.At(i, j)))
			if im > maxIm {
				maxIm = im
			}
			if math.Abs(re) > maxRe {
				maxRe = math.Abs(re)
			}
			result.Set(i, j, re)
		}
	}
	tol := 1e-8 * math.Max(1, maxRe)
	if maxIm > tol {
		return nil, fmt.Errorf("matLog: residual imaginary part %.3e exceeds tolerance %.3e (unpaired complex eigenvalues): %w",
			maxIm, tol, ErrSingularTransform)
	}
	return result, nil
}
