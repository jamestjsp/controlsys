package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestLoopsens_FirstOrder(t *testing.T) {
	L, err := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Loopsens(L)
	if err != nil {
		t.Fatal(err)
	}

	sDC, err := res.So.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(sDC.At(0, 0)-2.0/3.0) > 1e-8 {
		t.Errorf("dcgain(S) = %g, want 2/3", sDC.At(0, 0))
	}

	tDC, err := res.To.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(tDC.At(0, 0)-1.0/3.0) > 1e-8 {
		t.Errorf("dcgain(T) = %g, want 1/3", tDC.At(0, 0))
	}

	if math.Abs(sDC.At(0, 0)+tDC.At(0, 0)-1.0) > 1e-8 {
		t.Errorf("S+T DC gain = %g, want 1", sDC.At(0, 0)+tDC.At(0, 0))
	}

	for _, w := range []float64{0.01, 0.1, 1, 10, 100} {
		sResp, _ := res.So.EvalFr(complex(0, w))
		tResp, _ := res.To.EvalFr(complex(0, w))
		sum := sResp[0][0] + tResp[0][0]
		if cmplx.Abs(sum-1) > 1e-6 {
			t.Errorf("w=%g: S+T = %v, want 1", w, sum)
		}
	}
}

func TestLoopsens_Integrator(t *testing.T) {
	L, err := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Loopsens(L)
	if err != nil {
		t.Fatal(err)
	}

	tDC, err := res.To.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(tDC.At(0, 0)-1.0) > 1e-8 {
		t.Errorf("dcgain(T) = %g, want 1", tDC.At(0, 0))
	}

	for _, w := range []float64{0.1, 1, 10} {
		sResp, _ := res.So.EvalFr(complex(0, w))
		tResp, _ := res.To.EvalFr(complex(0, w))
		sum := sResp[0][0] + tResp[0][0]
		if cmplx.Abs(sum-1) > 1e-6 {
			t.Errorf("w=%g: S+T = %v, want 1", w, sum)
		}
	}
}

func TestLoopsens_NilError(t *testing.T) {
	_, err := Loopsens(nil)
	if err == nil {
		t.Error("expected error for nil input")
	}
}

func TestLoopsens_NonSquareError(t *testing.T) {
	L, err := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 1, []float64{0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	_, err = Loopsens(L)
	if err == nil {
		t.Error("expected error for non-square system")
	}
}

func TestLoopsens_StabilityCheck(t *testing.T) {
	L, err := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Loopsens(L)
	if err != nil {
		t.Fatal(err)
	}

	stable, err := res.So.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Error("sensitivity should be stable")
	}

	stable, err = res.To.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Error("complementary sensitivity should be stable")
	}
}
