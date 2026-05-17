package controlsys

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func solveIdentityMinusProduct(left, right *mat.Dense, size int, context string, singular error) (*mat.Dense, error) {
	return solveIdentityMinusScaledProduct(left, right, 1, size, context, singular)
}

func solveIdentityMinusScaledProduct(left, right *mat.Dense, scale float64, size int, context string, singular error) (*mat.Dense, error) {
	loop := mat.NewDense(size, size, nil)
	loop.Mul(left, right)
	if scale != 1 {
		loop.Scale(scale, loop)
	}
	eye := eyeDense(size)
	loop.Sub(eye, loop)

	var lu mat.LU
	lu.Factorize(loop)
	if luNearSingular(&lu) {
		return nil, fmt.Errorf("%s: direct feedthrough loop is singular: %w", context, singular)
	}

	result := mat.NewDense(size, size, nil)
	if err := lu.SolveTo(result, false, eye); err != nil {
		return nil, fmt.Errorf("%s: direct feedthrough loop solve failed: %w", context, singular)
	}
	return result, nil
}

func solveFeedbackFeedthrough(plantD, controllerD *mat.Dense, sign float64, size int, context string, singular error) (*mat.Dense, error) {
	return solveIdentityMinusScaledProduct(plantD, controllerD, sign, size, context, singular)
}
