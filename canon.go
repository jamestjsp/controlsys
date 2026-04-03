package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"sort"

	"gonum.org/v1/gonum/mat"
)

type CanonForm string

const (
	CanonModal     CanonForm = "modal"
	CanonCompanion CanonForm = "companion"
)

type CanonResult struct {
	Sys *System
	T   *mat.Dense
}

func Canon(sys *System, form CanonForm) (*CanonResult, error) {
	if sys.HasDelay() {
		return nil, fmt.Errorf("controlsys: Canon does not support delayed systems; use Pade/AbsorbDelay first")
	}
	switch form {
	case CanonModal:
		return canonModal(sys)
	case CanonCompanion:
		return canonCompanion(sys)
	default:
		return nil, fmt.Errorf("controlsys: unknown canonical form %q", form)
	}
}

type eigBlock struct {
	mag   float64
	real1 float64
	imag1 float64
	idx   int
}

func canonModal(sys *System) (*CanonResult, error) {
	n, m, p := sys.Dims()
	if n == 0 {
		cp := sys.Copy()
		return &CanonResult{Sys: cp, T: &mat.Dense{}}, nil
	}

	var eig mat.Eigen
	ok := eig.Factorize(sys.A, mat.EigenRight)
	if !ok {
		return nil, fmt.Errorf("controlsys: eigendecomposition failed")
	}

	vals := eig.Values(nil)
	var vecs mat.CDense
	eig.VectorsTo(&vecs)

	blocks := make([]eigBlock, 0, n)
	used := make([]bool, n)
	for i := 0; i < n; i++ {
		if used[i] {
			continue
		}
		eigMag := cmplx.Abs(vals[i])
		realTol := 1e-10 * math.Max(1, eigMag)
		conj := -1
		if imag(vals[i]) != 0 {
			for j := i + 1; j < n; j++ {
				if !used[j] && isConjugate(vals[i], vals[j]) {
					conj = j
					break
				}
			}
		}
		if conj >= 0 {
			blocks = append(blocks, eigBlock{
				mag:   eigMag,
				real1: real(vals[i]),
				imag1: math.Abs(imag(vals[i])),
				idx:   i,
			})
			used[i] = true
			used[conj] = true
		} else if math.Abs(imag(vals[i])) < realTol {
			blocks = append(blocks, eigBlock{
				mag:   math.Abs(real(vals[i])),
				real1: real(vals[i]),
				imag1: 0,
				idx:   i,
			})
			used[i] = true
		} else {
			return nil, fmt.Errorf("controlsys: complex eigenvalue without conjugate pair")
		}
	}

	sort.Slice(blocks, func(i, j int) bool {
		return blocks[i].mag < blocks[j].mag
	})

	tData := make([]float64, n*n)
	col := 0
	for _, blk := range blocks {
		if blk.imag1 == 0 {
			for row := 0; row < n; row++ {
				tData[row*n+col] = real(vecs.At(row, blk.idx))
			}
			col++
		} else {
			for row := 0; row < n; row++ {
				v := vecs.At(row, blk.idx)
				tData[row*n+col] = real(v)
				tData[row*n+col+1] = imag(v)
			}
			col += 2
		}
	}

	T := mat.NewDense(n, n, tData)

	var lu mat.LU
	lu.Factorize(T)
	if luNearSingular(&lu) {
		return nil, fmt.Errorf("controlsys: transformation matrix is singular: %w", ErrSingularTransform)
	}

	Tinv := mat.NewDense(n, n, nil)
	eye := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		eye.Set(i, i, 1)
	}
	if err := lu.SolveTo(Tinv, false, eye); err != nil {
		return nil, fmt.Errorf("controlsys: inversion failed: %w", ErrSingularTransform)
	}

	Anew := mat.NewDense(n, n, nil)
	tmp := mat.NewDense(n, n, nil)
	tmp.Mul(Tinv, sys.A)
	Anew.Mul(tmp, T)

	Bnew := mat.NewDense(n, m, nil)
	Bnew.Mul(Tinv, sys.B)

	Cnew := mat.NewDense(p, n, nil)
	Cnew.Mul(sys.C, T)

	Dnew := denseCopy(sys.D)

	newSys, err := newNoCopy(Anew, Bnew, Cnew, Dnew, sys.Dt)
	if err != nil {
		return nil, err
	}
	propagateIONames(newSys, sys)

	return &CanonResult{Sys: newSys, T: T}, nil
}

func canonCompanion(sys *System) (*CanonResult, error) {
	n, m, p := sys.Dims()
	if n == 0 {
		cp := sys.Copy()
		return &CanonResult{Sys: cp, T: &mat.Dense{}}, nil
	}
	if m != 1 || p != 1 {
		return nil, fmt.Errorf("controlsys: companion form requires SISO system: %w", ErrNotSISO)
	}

	// Observable companion form via similarity: T = Oc^{-1} * Ocomp
	// where Oc = obsv(A,C) and Ocomp = obsv(Ac, Cc) with Cc = e1'

	charPoly := characteristicPoly(sys.A, n)
	lead := charPoly[n]
	for i := range charPoly {
		charPoly[i] /= lead
	}

	// Build companion A: superdiagonal 1s, last row = -coeffs
	Ac := mat.NewDense(n, n, nil)
	for i := 0; i < n-1; i++ {
		Ac.Set(i, i+1, 1)
	}
	for j := 0; j < n; j++ {
		Ac.Set(n-1, j, -charPoly[j])
	}

	// Cc = [1 0 ... 0]
	Cc := mat.NewDense(1, n, nil)
	Cc.Set(0, 0, 1)

	Oc := buildObsvMatrix(sys.A, sys.C, n, p)
	Ocomp := buildObsvMatrix(Ac, Cc, n, 1)

	var luOc mat.LU
	luOc.Factorize(Oc)
	if luNearSingular(&luOc) {
		return nil, fmt.Errorf("controlsys: system not observable, companion form undefined: %w", ErrSingularTransform)
	}

	// T = Oc \ Ocomp  (i.e. Oc * T = Ocomp => T = Oc^{-1} * Ocomp)
	T := mat.NewDense(n, n, nil)
	if err := luOc.SolveTo(T, false, Ocomp); err != nil {
		return nil, fmt.Errorf("controlsys: transformation solve failed: %w", ErrSingularTransform)
	}

	var luT mat.LU
	luT.Factorize(T)
	if luNearSingular(&luT) {
		return nil, fmt.Errorf("controlsys: transformation matrix singular: %w", ErrSingularTransform)
	}

	Tinv := mat.NewDense(n, n, nil)
	eye := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		eye.Set(i, i, 1)
	}
	if err := luT.SolveTo(Tinv, false, eye); err != nil {
		return nil, fmt.Errorf("controlsys: inversion failed: %w", ErrSingularTransform)
	}

	Anew := mat.NewDense(n, n, nil)
	tmp := mat.NewDense(n, n, nil)
	tmp.Mul(Tinv, sys.A)
	Anew.Mul(tmp, T)

	Bnew := mat.NewDense(n, 1, nil)
	Bnew.Mul(Tinv, sys.B)

	Cnew := mat.NewDense(1, n, nil)
	Cnew.Mul(sys.C, T)

	Dnew := denseCopy(sys.D)

	newSys, err := newNoCopy(Anew, Bnew, Cnew, Dnew, sys.Dt)
	if err != nil {
		return nil, err
	}
	propagateIONames(newSys, sys)

	return &CanonResult{Sys: newSys, T: T}, nil
}

func characteristicPoly(A *mat.Dense, n int) []float64 {
	if n == 0 {
		return []float64{1}
	}

	var eig mat.Eigen
	ok := eig.Factorize(A, mat.EigenNone)
	if !ok {
		coeffs := make([]float64, n+1)
		coeffs[n] = 1
		return coeffs
	}
	vals := eig.Values(nil)

	// Build polynomial from roots incrementally: prod(s - lambda_i)
	// coeffs in descending power: coeffs[0]*s^n + ... + coeffs[n]
	coeffs := make([]float64, n+1)
	coeffs[0] = 1.0
	cur := 1
	for _, lam := range vals {
		if math.Abs(imag(lam)) < 1e-10*math.Max(1, cmplx.Abs(lam)) {
			r := real(lam)
			for j := cur; j >= 1; j-- {
				coeffs[j] = coeffs[j-1] - r*coeffs[j]
			}
			coeffs[0] = -r * coeffs[0]
			cur++
		} else if imag(lam) > 0 {
			a := -2 * real(lam)
			b := real(lam)*real(lam) + imag(lam)*imag(lam)
			newCoeffs := make([]float64, n+1)
			for j := 0; j <= cur; j++ {
				newCoeffs[j+2] += coeffs[j]
				newCoeffs[j+1] += a * coeffs[j]
				newCoeffs[j] += b * coeffs[j]
			}
			copy(coeffs, newCoeffs)
			cur += 2
		}
	}

	// Result in ascending power order: result[i] = coefficient of s^i
	result := make([]float64, n+1)
	for i := range result {
		result[i] = coeffs[n-i]
	}
	return result
}

func buildObsvMatrix(A, C *mat.Dense, n, p int) *mat.Dense {
	O := mat.NewDense(n*p, n, nil)
	setBlock(O, 0, 0, C)
	CA := mat.NewDense(p, n, nil)
	CA.Mul(C, A)
	prev := CA
	setBlock(O, p, 0, CA)
	for i := 2; i < n; i++ {
		next := mat.NewDense(p, n, nil)
		next.Mul(prev, A)
		setBlock(O, i*p, 0, next)
		prev = next
	}

	if n*p == n {
		return O
	}
	return extractSubmatrix(O, 0, n, 0, n)
}
