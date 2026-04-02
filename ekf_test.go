package controlsys

import (
	"math"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestEKF_LinearSystem(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.95})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})

	qScale := 0.01
	rScale := 0.1
	Q := mat.NewDense(2, 2, []float64{qScale, 0, 0, qScale})
	R := mat.NewDense(1, 1, []float64{rScale})

	model := &EKFModel{
		F: func(x, u *mat.VecDense) *mat.VecDense {
			out := mat.NewVecDense(2, nil)
			out.MulVec(A, x)
			var bu mat.VecDense
			bu.MulVec(B, u)
			out.AddVec(out, &bu)
			return out
		},
		H: func(x *mat.VecDense) *mat.VecDense {
			out := mat.NewVecDense(1, nil)
			out.MulVec(C, x)
			return out
		},
		FJac: func(x, u *mat.VecDense) *mat.Dense { return mat.DenseCopyOf(A) },
		HJac: func(x *mat.VecDense) *mat.Dense { return mat.DenseCopyOf(C) },
		Q:    Q,
		R:    R,
	}

	x0 := mat.NewVecDense(2, []float64{0, 0})
	P0 := mat.NewDense(2, 2, []float64{1, 0, 0, 1})

	ekf, err := NewEKF(model, x0, P0)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(42, 0))
	xTrue := mat.NewVecDense(2, []float64{1, 0.5})
	u := mat.NewVecDense(1, []float64{0.1})

	nSteps := 50
	for k := range nSteps {
		wNoise := mat.NewVecDense(2, []float64{
			rng.NormFloat64() * math.Sqrt(qScale),
			rng.NormFloat64() * math.Sqrt(qScale),
		})
		xNext := mat.NewVecDense(2, nil)
		xNext.MulVec(A, xTrue)
		var bu mat.VecDense
		bu.MulVec(B, u)
		xNext.AddVec(xNext, &bu)
		xNext.AddVec(xNext, wNoise)

		vNoise := mat.NewVecDense(1, []float64{rng.NormFloat64() * math.Sqrt(rScale)})
		z := mat.NewVecDense(1, nil)
		z.MulVec(C, xNext)
		z.AddVec(z, vNoise)

		xTrue = xNext

		if err := ekf.Step(u, z); err != nil {
			t.Fatalf("step %d: %v", k, err)
		}
	}

	for i := range 2 {
		diff := math.Abs(ekf.X.AtVec(i) - xTrue.AtVec(i))
		if diff > 1.0 {
			t.Errorf("state %d: |ekf-true| = %v, want < 1.0", i, diff)
		}
	}
}

func TestEKF_NonlinearPendulum(t *testing.T) {
	const (
		g  = 9.81
		L  = 1.0
		dt = 0.01
	)

	model := &EKFModel{
		F: func(x, u *mat.VecDense) *mat.VecDense {
			theta, omega := x.AtVec(0), x.AtVec(1)
			return mat.NewVecDense(2, []float64{
				theta + dt*omega,
				omega + dt*(-g/L*math.Sin(theta)),
			})
		},
		H: func(x *mat.VecDense) *mat.VecDense {
			return mat.NewVecDense(1, []float64{x.AtVec(0)})
		},
		FJac: func(x, u *mat.VecDense) *mat.Dense {
			theta := x.AtVec(0)
			return mat.NewDense(2, 2, []float64{
				1, dt,
				-dt * g / L * math.Cos(theta), 1,
			})
		},
		HJac: func(x *mat.VecDense) *mat.Dense {
			return mat.NewDense(1, 2, []float64{1, 0})
		},
		Q: mat.NewDense(2, 2, []float64{1e-4, 0, 0, 1e-4}),
		R: mat.NewDense(1, 1, []float64{0.01}),
	}

	x0 := mat.NewVecDense(2, []float64{0, 0})
	P0 := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	ekf, err := NewEKF(model, x0, P0)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(99, 0))
	xTrue := mat.NewVecDense(2, []float64{0.3, 0})
	noU := mat.NewVecDense(1, nil)

	nSteps := 100
	for k := range nSteps {
		theta, omega := xTrue.AtVec(0), xTrue.AtVec(1)
		xTrue = mat.NewVecDense(2, []float64{
			theta + dt*omega + rng.NormFloat64()*1e-2,
			omega + dt*(-g/L*math.Sin(theta)) + rng.NormFloat64()*1e-2,
		})

		z := mat.NewVecDense(1, []float64{xTrue.AtVec(0) + rng.NormFloat64()*0.1})

		if err := ekf.Step(noU, z); err != nil {
			t.Fatalf("step %d: %v", k, err)
		}
	}

	diff := math.Abs(ekf.X.AtVec(0) - xTrue.AtVec(0))
	if diff > 0.5 {
		t.Errorf("|theta_ekf - theta_true| = %v, want < 0.5", diff)
	}
}

func TestEKF_StepEquivalence(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.95})
	C := mat.NewDense(1, 2, []float64{1, 0})
	Q := mat.NewDense(2, 2, []float64{0.01, 0, 0, 0.01})
	R := mat.NewDense(1, 1, []float64{0.1})

	makeModel := func() *EKFModel {
		return &EKFModel{
			F: func(x, u *mat.VecDense) *mat.VecDense {
				out := mat.NewVecDense(2, nil)
				out.MulVec(A, x)
				return out
			},
			H: func(x *mat.VecDense) *mat.VecDense {
				out := mat.NewVecDense(1, nil)
				out.MulVec(C, x)
				return out
			},
			FJac: func(x, u *mat.VecDense) *mat.Dense { return mat.DenseCopyOf(A) },
			HJac: func(x *mat.VecDense) *mat.Dense { return mat.DenseCopyOf(C) },
			Q:    Q,
			R:    R,
		}
	}

	x0 := mat.NewVecDense(2, []float64{1, 2})
	P0 := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	u := mat.NewVecDense(1, nil)
	z := mat.NewVecDense(1, []float64{0.8})

	ekf1, _ := NewEKF(makeModel(), x0, P0)
	ekf2, _ := NewEKF(makeModel(), x0, P0)

	if err := ekf1.Predict(u); err != nil {
		t.Fatal(err)
	}
	if err := ekf1.Update(z); err != nil {
		t.Fatal(err)
	}
	if err := ekf2.Step(u, z); err != nil {
		t.Fatal(err)
	}

	for i := range 2 {
		if math.Abs(ekf1.X.AtVec(i)-ekf2.X.AtVec(i)) > 1e-14 {
			t.Errorf("X[%d]: predict+update=%v step=%v", i, ekf1.X.AtVec(i), ekf2.X.AtVec(i))
		}
	}
	r1, c1 := ekf1.P.Dims()
	for i := range r1 {
		for j := range c1 {
			if math.Abs(ekf1.P.At(i, j)-ekf2.P.At(i, j)) > 1e-14 {
				t.Errorf("P[%d,%d]: predict+update=%v step=%v", i, j, ekf1.P.At(i, j), ekf2.P.At(i, j))
			}
		}
	}
}

func TestEKF_CovariancePSD(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.95})
	C := mat.NewDense(1, 2, []float64{1, 0})

	model := &EKFModel{
		F: func(x, u *mat.VecDense) *mat.VecDense {
			out := mat.NewVecDense(2, nil)
			out.MulVec(A, x)
			return out
		},
		H: func(x *mat.VecDense) *mat.VecDense {
			out := mat.NewVecDense(1, nil)
			out.MulVec(C, x)
			return out
		},
		FJac: func(x, u *mat.VecDense) *mat.Dense { return mat.DenseCopyOf(A) },
		HJac: func(x *mat.VecDense) *mat.Dense { return mat.DenseCopyOf(C) },
		Q:    mat.NewDense(2, 2, []float64{0.01, 0, 0, 0.01}),
		R:    mat.NewDense(1, 1, []float64{0.1}),
	}

	ekf, _ := NewEKF(model, mat.NewVecDense(2, []float64{1, 0}), mat.NewDense(2, 2, []float64{1, 0, 0, 1}))
	rng := rand.New(rand.NewPCG(7, 0))
	u := mat.NewVecDense(1, nil)

	for k := range 30 {
		z := mat.NewVecDense(1, []float64{rng.NormFloat64()})
		if err := ekf.Step(u, z); err != nil {
			t.Fatalf("step %d: %v", k, err)
		}

		n, _ := ekf.P.Dims()
		for i := range n {
			for j := range n {
				if math.Abs(ekf.P.At(i, j)-ekf.P.At(j, i)) > 1e-12 {
					t.Fatalf("step %d: P not symmetric: P[%d,%d]=%v P[%d,%d]=%v",
						k, i, j, ekf.P.At(i, j), j, i, ekf.P.At(j, i))
				}
			}
		}

		var eig mat.Eigen
		ok := eig.Factorize(ekf.P, mat.EigenNone)
		if !ok {
			t.Fatalf("step %d: eigen decomposition failed", k)
		}
		for _, ev := range eig.Values(nil) {
			if real(ev) < -1e-10 {
				t.Fatalf("step %d: negative eigenvalue %v", k, ev)
			}
		}
	}
}

func TestEKF_Validation(t *testing.T) {
	dummyF := func(x, u *mat.VecDense) *mat.VecDense { return x }
	dummyH := func(x *mat.VecDense) *mat.VecDense { return mat.NewVecDense(1, nil) }
	dummyFJac := func(x, u *mat.VecDense) *mat.Dense { return mat.NewDense(2, 2, nil) }
	dummyHJac := func(x *mat.VecDense) *mat.Dense { return mat.NewDense(1, 2, nil) }

	x0 := mat.NewVecDense(2, nil)
	P0 := mat.NewDense(2, 2, nil)
	Q := mat.NewDense(2, 2, nil)
	R := mat.NewDense(1, 1, nil)

	t.Run("nil model", func(t *testing.T) {
		_, err := NewEKF(nil, x0, P0)
		if err == nil {
			t.Fatal("expected error for nil model")
		}
	})

	t.Run("nil function", func(t *testing.T) {
		_, err := NewEKF(&EKFModel{F: dummyF, H: dummyH, FJac: nil, HJac: dummyHJac, Q: Q, R: R}, x0, P0)
		if err == nil {
			t.Fatal("expected error for nil FJac")
		}
	})

	t.Run("P0 dimension mismatch", func(t *testing.T) {
		badP := mat.NewDense(3, 3, nil)
		m := &EKFModel{F: dummyF, H: dummyH, FJac: dummyFJac, HJac: dummyHJac, Q: Q, R: R}
		_, err := NewEKF(m, x0, badP)
		if err == nil {
			t.Fatal("expected error for P0 size mismatch")
		}
	})

	t.Run("Q dimension mismatch", func(t *testing.T) {
		badQ := mat.NewDense(3, 3, nil)
		m := &EKFModel{F: dummyF, H: dummyH, FJac: dummyFJac, HJac: dummyHJac, Q: badQ, R: R}
		_, err := NewEKF(m, x0, P0)
		if err == nil {
			t.Fatal("expected error for Q size mismatch")
		}
	})

	t.Run("nil x0", func(t *testing.T) {
		m := &EKFModel{F: dummyF, H: dummyH, FJac: dummyFJac, HJac: dummyHJac, Q: Q, R: R}
		_, err := NewEKF(m, nil, P0)
		if err == nil {
			t.Fatal("expected error for nil x0")
		}
	})
}
