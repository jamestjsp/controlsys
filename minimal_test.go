package controlsys

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func markovParams(sys *System, k int) []float64 {
	n, m, p := sys.Dims()
	if k == 0 {
		if p == 0 || m == 0 {
			return nil
		}
		data := make([]float64, p*m)
		for i := 0; i < p; i++ {
			for j := 0; j < m; j++ {
				data[i*m+j] = sys.D.At(i, j)
			}
		}
		return data
	}
	if n == 0 || p == 0 || m == 0 {
		return make([]float64, p*m)
	}
	// C * A^(k-1) * B
	pow := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		pow.Set(i, i, 1)
	}
	for i := 0; i < k-1; i++ {
		var tmp mat.Dense
		tmp.Mul(pow, sys.A)
		pow.Copy(&tmp)
	}
	var cab mat.Dense
	var tmp2 mat.Dense
	tmp2.Mul(sys.C, pow)
	cab.Mul(&tmp2, sys.B)
	r, c := cab.Dims()
	data := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[i*c+j] = cab.At(i, j)
		}
	}
	return data
}

func checkMarkovEqual(t *testing.T, orig, reduced *System, maxK int, tol float64) {
	t.Helper()
	for k := 0; k <= maxK; k++ {
		mo := markovParams(orig, k)
		mr := markovParams(reduced, k)
		if len(mo) != len(mr) {
			t.Errorf("Markov param k=%d: length mismatch %d vs %d", k, len(mo), len(mr))
			continue
		}
		for i := range mo {
			if math.Abs(mo[i]-mr[i]) > tol {
				t.Errorf("Markov param k=%d, elem %d: orig=%g reduced=%g", k, i, mo[i], mr[i])
			}
		}
	}
}

func TestReduceAlreadyMinimal(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}

	if res.Order != 2 {
		t.Errorf("expected order 2, got %d", res.Order)
	}

	checkMarkovEqual(t, sys, res.Sys, 6, 1e-10)
}

func TestReduceUncontrollableBZero(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{1, 0, 0, 0, 2, 0, 0, 0, 3})
	B := mat.NewDense(3, 1, nil)
	C := mat.NewDense(1, 3, []float64{1, 1, 1})
	D := mat.NewDense(1, 1, []float64{5})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}

	if res.Order != 0 {
		t.Errorf("expected order 0 for B=0, got %d", res.Order)
	}

	if res.Sys.D.At(0, 0) != 5 {
		t.Errorf("D not preserved: got %g", res.Sys.D.At(0, 0))
	}
}

func TestReduceUnobservableCZero(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{1, 0, 0, 0, 2, 0, 0, 0, 3})
	B := mat.NewDense(3, 1, []float64{1, 1, 1})
	C := mat.NewDense(1, 3, nil)
	D := mat.NewDense(1, 1, []float64{7})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}

	if res.Order != 0 {
		t.Errorf("expected order 0 for C=0, got %d", res.Order)
	}

	if res.Sys.D.At(0, 0) != 7 {
		t.Errorf("D not preserved: got %g", res.Sys.D.At(0, 0))
	}
}

func TestReducePartiallyControllableObservable(t *testing.T) {
	// 4-state system: states 1,2 controllable+observable; state 3 uncontrollable; state 4 unobservable
	A := mat.NewDense(4, 4, []float64{
		0, 1, 0, 0,
		-2, -3, 0, 0,
		0, 0, -5, 0,
		0, 0, 0, -7,
	})
	B := mat.NewDense(4, 1, []float64{0, 1, 0, 1})
	C := mat.NewDense(1, 4, []float64{1, 0, 0, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}

	if res.Order > 3 {
		t.Errorf("expected order <= 3, got %d", res.Order)
	}

	n, _, _ := sys.Dims()
	checkMarkovEqual(t, sys, res.Sys, 2*n, 1e-8)
}

func TestReduceUncontrollableOnly(t *testing.T) {
	// State 3 is uncontrollable but observable
	A := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		-2, -3, 0,
		0, 0, -5,
	})
	B := mat.NewDense(3, 1, []float64{0, 1, 0})
	C := mat.NewDense(1, 3, []float64{1, 0, 1})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := sys.Reduce(&ReduceOpts{Mode: ReduceUncontrollable})
	if err != nil {
		t.Fatal(err)
	}

	if res.Order != 2 {
		t.Errorf("expected order 2 (controllable only), got %d", res.Order)
	}
}

func TestReduceUnobservableOnly(t *testing.T) {
	// State 3 is controllable but unobservable
	A := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		-2, -3, 0,
		0, 0, -5,
	})
	B := mat.NewDense(3, 1, []float64{0, 1, 1})
	C := mat.NewDense(1, 3, []float64{1, 0, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := sys.Reduce(&ReduceOpts{Mode: ReduceUnobservable})
	if err != nil {
		t.Fatal(err)
	}

	if res.Order != 2 {
		t.Errorf("expected order 2 (observable only), got %d", res.Order)
	}
}

func TestReduceZeroDimension(t *testing.T) {
	sys := &System{A: &mat.Dense{}, B: &mat.Dense{}, C: &mat.Dense{}, D: &mat.Dense{}}

	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}

	if res.Order != 0 {
		t.Errorf("expected order 0, got %d", res.Order)
	}
}

func TestMinimalRealizationAlias(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := sys.MinimalRealization()
	if err != nil {
		t.Fatal(err)
	}

	if res.Order != 2 {
		t.Errorf("expected order 2, got %d", res.Order)
	}
}

func TestReduceMarkovPreserved(t *testing.T) {
	// Controllable canonical form for (s+1)(s+2)(s+3)
	A := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		0, 0, 1,
		-6, -11, -6,
	})
	B := mat.NewDense(3, 1, []float64{0, 0, 1})
	C := mat.NewDense(1, 3, []float64{1, 0, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := sys.Dims()
	checkMarkovEqual(t, sys, res.Sys, 2*n, 1e-10)
}

func TestReduceMZero(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0, 0, 2})
	B := &mat.Dense{}
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := &mat.Dense{}

	sys := &System{A: A, B: B, C: C, D: D}

	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}

	if res.Order != 0 {
		t.Errorf("expected order 0 for m=0, got %d", res.Order)
	}
}

func TestReducePZero(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0, 0, 2})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := &mat.Dense{}
	D := &mat.Dense{}

	sys := &System{A: A, B: B, C: C, D: D}

	res, err := sys.Reduce(nil)
	if err != nil {
		t.Fatal(err)
	}

	if res.Order != 0 {
		t.Errorf("expected order 0 for p=0, got %d", res.Order)
	}
}
