package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func evalTF(sys *System, s complex128) [][]complex128 {
	tf, err := sys.TransferFunction(nil)
	if err != nil {
		panic(err)
	}
	return tf.TF.Eval(s)
}

func assertTFClose(t *testing.T, sys *System, s complex128, want complex128, tol float64) {
	t.Helper()
	h := evalTF(sys, s)
	got := h[0][0]
	if cmplx.Abs(got-want) > tol {
		t.Errorf("H(%v) = %v, want %v (diff=%v)", s, got, want, cmplx.Abs(got-want))
	}
}

func TestSeries_SISO(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2, _ := NewGain(mat.NewDense(1, 1, []float64{2}), 0)

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := result.Dims()
	if n != 1 || m != 1 || p != 1 {
		t.Fatalf("dims = (%d,%d,%d), want (1,1,1)", n, m, p)
	}

	for _, omega := range []float64{0.1, 1.0, 10.0} {
		s := complex(0, omega)
		want := 2.0 / s
		assertTFClose(t, result, s, want, 1e-10)
	}
}

func TestSeries_MIMO(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(2, 1, []float64{1, -1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := result.Dims()
	if n != 3 {
		t.Errorf("n = %d, want 3", n)
	}
	if m != 2 || p != 2 {
		t.Errorf("m,p = %d,%d, want 2,2", m, p)
	}

	s := complex(0, 1.0)
	h1 := evalTF(sys1, s)
	h2 := evalTF(sys2, s)
	hc := evalTF(result, s)

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			want := h2[i][0]*h1[0][j] + h2[i][1]*h1[1][j]
			if cmplx.Abs(hc[i][j]-want) > 1e-8 {
				t.Errorf("H[%d][%d] at s=j: got %v, want %v", i, j, hc[i][j], want)
			}
		}
	}
}

func TestSeries_DimMismatch(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(2, 1, []float64{0, 0}),
		0,
	)
	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_, err := Series(sys1, sys2)
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

func TestSeries_DomainMismatch(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.01,
	)
	_, err := Series(sys1, sys2)
	if !errors.Is(err, ErrDomainMismatch) {
		t.Errorf("expected ErrDomainMismatch, got %v", err)
	}
}

func TestSeries_GainSystems(t *testing.T) {
	g1, _ := NewGain(mat.NewDense(1, 1, []float64{3}), 0)
	g2, _ := NewGain(mat.NewDense(1, 1, []float64{5}), 0)

	result, err := Series(g1, g2)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := result.Dims()
	if n != 0 {
		t.Errorf("n = %d, want 0", n)
	}
	if result.D.At(0, 0) != 15 {
		t.Errorf("D = %v, want 15", result.D.At(0, 0))
	}
}

func TestSeries_Roundtrip(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}

	for _, omega := range []float64{0.1, 0.5, 1.0, 5.0, 10.0} {
		s := complex(0, omega)
		h1 := evalTF(sys1, s)[0][0]
		h2 := evalTF(sys2, s)[0][0]
		hc := evalTF(result, s)[0][0]
		want := h2 * h1
		if cmplx.Abs(hc-want) > 1e-10 {
			t.Errorf("w=%v: got %v, want %v", omega, hc, want)
		}
	}
}

func TestParallel_SISO(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	result, err := Parallel(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := result.Dims()
	if n != 2 {
		t.Fatalf("n = %d, want 2", n)
	}

	for _, omega := range []float64{0.1, 1.0, 10.0} {
		s := complex(0, omega)
		h1 := evalTF(sys1, s)[0][0]
		h2 := evalTF(sys2, s)[0][0]
		hc := evalTF(result, s)[0][0]
		want := h1 + h2
		if cmplx.Abs(hc-want) > 1e-10 {
			t.Errorf("w=%v: got %v, want %v", omega, hc, want)
		}
	}
}

func TestParallel_DimMismatch(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 2, []float64{0, 0}),
		0,
	)
	_, err := Parallel(sys1, sys2)
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

func TestSeriesDelay_SISO(t *testing.T) {
	sys1, _ := NewFromSlices(1, 1, 1, []float64{0.5}, []float64{1}, []float64{1}, []float64{0}, 1.0)
	sys1.Delay = mat.NewDense(1, 1, []float64{2})
	sys2, _ := NewFromSlices(1, 1, 1, []float64{0.8}, []float64{1}, []float64{1}, []float64{0}, 1.0)
	sys2.Delay = mat.NewDense(1, 1, []float64{3})

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if result.Delay == nil {
		t.Fatal("series should preserve SISO delay")
	}
	if result.Delay.At(0, 0) != 5 {
		t.Errorf("delay = %v, want 5 (2+3)", result.Delay.At(0, 0))
	}
}

func TestSeriesDelay_MIMO_DifferentPaths(t *testing.T) {
	// Paths through different intermediate channels have different total delays
	sys1, _ := NewFromSlices(1, 2, 2,
		[]float64{0.5}, []float64{1, 0}, []float64{1, 1}, []float64{0, 0, 0, 0}, 1.0)
	sys1.Delay = mat.NewDense(2, 2, []float64{
		2, 3,
		2, 3,
	})
	sys2, _ := NewFromSlices(1, 2, 1,
		[]float64{0.8}, []float64{1, 1}, []float64{1}, []float64{0, 0}, 1.0)
	sys2.Delay = mat.NewDense(1, 2, []float64{1, 4})

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	// d1[0,0]+d2[0,0]=3 vs d1[1,0]+d2[0,1]=6 → different → nil
	if result.Delay != nil {
		t.Error("different path delays should yield nil delay")
	}
}

func TestSeriesDelay_MIMO_NonUniform(t *testing.T) {
	// Different path delays → should return nil
	sys1, _ := NewFromSlices(1, 2, 2,
		[]float64{0.5}, []float64{1, 0}, []float64{1, 1}, []float64{0, 0, 0, 0}, 1.0)
	sys1.Delay = mat.NewDense(2, 2, []float64{
		2, 3,
		5, 3,
	})
	sys2, _ := NewFromSlices(1, 2, 1,
		[]float64{0.8}, []float64{1, 1}, []float64{1}, []float64{0, 0}, 1.0)
	sys2.Delay = mat.NewDense(1, 2, []float64{1, 1})

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	// d1[0,0]+d2[0,0]=3 vs d1[1,0]+d2[0,1]=6 → non-uniform → nil
	if result.Delay != nil {
		t.Error("non-uniform path delays should yield nil delay")
	}
}

func TestSeriesDelay_MIMO_ConsistentPaths(t *testing.T) {
	// Delays chosen so all paths sum to same value per (i,j)
	sys1, _ := NewFromSlices(1, 1, 2,
		[]float64{0.5}, []float64{1}, []float64{1, 1}, []float64{0, 0}, 1.0)
	sys1.Delay = mat.NewDense(2, 1, []float64{3, 5})

	sys2, _ := NewFromSlices(1, 2, 1,
		[]float64{0.8}, []float64{1, 1}, []float64{1}, []float64{0, 0}, 1.0)
	sys2.Delay = mat.NewDense(1, 2, []float64{7, 5})

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	// Path through k=0: d1[0,0]+d2[0,0] = 3+7 = 10
	// Path through k=1: d1[1,0]+d2[0,1] = 5+5 = 10 ✓
	if result.Delay == nil {
		t.Fatal("consistent paths should preserve delay")
	}
	if result.Delay.At(0, 0) != 10 {
		t.Errorf("delay = %v, want 10", result.Delay.At(0, 0))
	}
}

func TestFeedback_Integrator(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)

	result, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}

	poles, err := result.Poles()
	if err != nil {
		t.Fatal(err)
	}
	if len(poles) != 1 {
		t.Fatalf("expected 1 pole, got %d", len(poles))
	}
	if math.Abs(real(poles[0])-(-1)) > 1e-10 || math.Abs(imag(poles[0])) > 1e-10 {
		t.Errorf("pole = %v, want -1", poles[0])
	}
}

func TestFeedback_SingularM(t *testing.T) {
	plant, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{-1}), 0)

	_, err := Feedback(plant, controller, -1)
	if !errors.Is(err, ErrSingularTransform) {
		t.Errorf("expected ErrSingularTransform, got %v", err)
	}
}

func TestFeedback_ZeroD(t *testing.T) {
	plant, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	controller, _ := New(
		mat.NewDense(1, 1, []float64{-5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	result, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := result.Dims()
	if n != 3 || m != 1 || p != 1 {
		t.Errorf("dims = (%d,%d,%d), want (3,1,1)", n, m, p)
	}
	stable, err := result.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Log("closed-loop system may not be stable with this controller")
	}
}

func TestFeedback_WithDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	plant.Delay = mat.NewDense(1, 1, []float64{0.5})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !cl.HasInternalDelay() {
		t.Error("closed-loop should have internal delays")
	}
	if len(cl.InternalDelay) != 1 || cl.InternalDelay[0] != 0.5 {
		t.Errorf("InternalDelay = %v, want [0.5]", cl.InternalDelay)
	}
}

func TestAppend_Basic(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	result, err := Append(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := result.Dims()
	if n != 2 || m != 2 || p != 2 {
		t.Fatalf("dims = (%d,%d,%d), want (2,2,2)", n, m, p)
	}

	if result.A.At(0, 1) != 0 || result.A.At(1, 0) != 0 {
		t.Error("A should be block-diagonal")
	}
	if result.B.At(0, 1) != 0 || result.B.At(1, 0) != 0 {
		t.Error("B should be block-diagonal")
	}
	if result.C.At(0, 1) != 0 || result.C.At(1, 0) != 0 {
		t.Error("C should be block-diagonal")
	}
}

func TestAppend_WithInputOutputDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys1.InputDelay = []float64{2}
	sys1.OutputDelay = []float64{1}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 2, []float64{0, 0}),
		0,
	)
	sys2.InputDelay = []float64{0, 3}
	sys2.OutputDelay = []float64{4}

	result, err := Append(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := result.Dims()
	if n != 2 || m != 3 || p != 2 {
		t.Fatalf("dims = (%d,%d,%d), want (2,3,2)", n, m, p)
	}
	wantIn := []float64{2, 0, 3}
	wantOut := []float64{1, 4}
	for i, w := range wantIn {
		if result.InputDelay[i] != w {
			t.Errorf("InputDelay[%d] = %v, want %v", i, result.InputDelay[i], w)
		}
	}
	for i, w := range wantOut {
		if result.OutputDelay[i] != w {
			t.Errorf("OutputDelay[%d] = %v, want %v", i, result.OutputDelay[i], w)
		}
	}
}

func TestAppend_WithMixedNilDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys1.InputDelay = []float64{5}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 2, []float64{0, 0}),
		0,
	)

	result, err := Append(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	wantIn := []float64{5, 0, 0}
	if len(result.InputDelay) != 3 {
		t.Fatalf("InputDelay length = %d, want 3", len(result.InputDelay))
	}
	for i, w := range wantIn {
		if result.InputDelay[i] != w {
			t.Errorf("InputDelay[%d] = %v, want %v", i, result.InputDelay[i], w)
		}
	}
	if result.OutputDelay != nil {
		t.Errorf("OutputDelay should be nil, got %v", result.OutputDelay)
	}
}

func TestSeries_WithInputOutputDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys1.InputDelay = []float64{2}
	sys1.OutputDelay = []float64{1}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2.InputDelay = []float64{3}
	sys2.OutputDelay = []float64{4}

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.InputDelay) != 1 || result.InputDelay[0] != 2 {
		t.Errorf("InputDelay = %v, want [2]", result.InputDelay)
	}
	if len(result.OutputDelay) != 1 || result.OutputDelay[0] != 4 {
		t.Errorf("OutputDelay = %v, want [4]", result.OutputDelay)
	}
	td := result.TotalDelay()
	if td == nil {
		t.Fatal("TotalDelay should not be nil")
	}
	if td.At(0, 0) != 2+1+3+4 {
		t.Errorf("TotalDelay = %v, want %v", td.At(0, 0), 2+1+3+4)
	}
}

func TestSeries_WithOnlyInputDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys1.InputDelay = []float64{5}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.InputDelay) != 1 || result.InputDelay[0] != 5 {
		t.Errorf("InputDelay = %v, want [5]", result.InputDelay)
	}
	if result.OutputDelay != nil {
		t.Errorf("OutputDelay should be nil, got %v", result.OutputDelay)
	}
}

func TestSeries_WithOnlyOutputDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2.OutputDelay = []float64{7}

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if result.InputDelay != nil {
		t.Errorf("InputDelay should be nil, got %v", result.InputDelay)
	}
	if len(result.OutputDelay) != 1 || result.OutputDelay[0] != 7 {
		t.Errorf("OutputDelay = %v, want [7]", result.OutputDelay)
	}
}

func TestSeries_IntermediateDelayUniform(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	sys1.OutputDelay = []float64{1, 1}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 2, []float64{0, 0}),
		0,
	)
	sys2.InputDelay = []float64{3, 3}

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	td := result.TotalDelay()
	if td == nil {
		t.Fatal("TotalDelay should not be nil")
	}
	for j := 0; j < 2; j++ {
		if td.At(0, j) != 4 {
			t.Errorf("TotalDelay[0][%d] = %v, want 4", j, td.At(0, j))
		}
	}
}

func TestSeries_IntermediateDelayNonUniform(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(2, 1, []float64{0, 0}),
		0,
	)
	sys1.OutputDelay = []float64{1, 3}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 2, []float64{0, 0}),
		0,
	)
	sys2.InputDelay = []float64{2, 4}

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if !result.HasInternalDelay() {
		t.Fatal("non-uniform intermediate delays should produce InternalDelay")
	}
	if len(result.InternalDelay) != 4 {
		t.Errorf("InternalDelay count = %d, want 4 (2 output + 2 input)", len(result.InternalDelay))
	}
}

func TestParallel_WithMatchingInputOutputDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys1.InputDelay = []float64{2}
	sys1.OutputDelay = []float64{3}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2.InputDelay = []float64{2}
	sys2.OutputDelay = []float64{3}

	result, err := Parallel(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.InputDelay) != 1 || result.InputDelay[0] != 2 {
		t.Errorf("InputDelay = %v, want [2]", result.InputDelay)
	}
	if len(result.OutputDelay) != 1 || result.OutputDelay[0] != 3 {
		t.Errorf("OutputDelay = %v, want [3]", result.OutputDelay)
	}
}

func TestParallel_WithMismatchedInputDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys1.InputDelay = []float64{1}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2.InputDelay = []float64{3}

	result, err := Parallel(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if !result.HasInternalDelay() {
		t.Fatal("mismatched input delays should produce InternalDelay")
	}
	if len(result.InternalDelay) != 1 {
		t.Errorf("InternalDelay count = %d, want 1 (only difference goes internal)", len(result.InternalDelay))
	}
	if result.InputDelay == nil || result.InputDelay[0] != 1 {
		t.Errorf("common InputDelay should stay external: got %v, want [1]", result.InputDelay)
	}
}

func TestParallel_WithMismatchedOutputDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys1.OutputDelay = []float64{2}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2.OutputDelay = []float64{5}

	result, err := Parallel(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if !result.HasInternalDelay() {
		t.Fatal("mismatched output delays should produce InternalDelay")
	}
	if len(result.InternalDelay) != 1 {
		t.Errorf("InternalDelay count = %d, want 1 (only difference goes internal)", len(result.InternalDelay))
	}
	if result.OutputDelay == nil || result.OutputDelay[0] != 2 {
		t.Errorf("common OutputDelay should stay external: got %v, want [2]", result.OutputDelay)
	}
}

func TestAppend_WithDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys1.Delay = mat.NewDense(1, 1, []float64{0.5})

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2.Delay = mat.NewDense(1, 1, []float64{1.0})

	result, err := Append(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if result.Delay == nil {
		t.Fatal("expected delay matrix")
	}
	r, c := result.Delay.Dims()
	if r != 2 || c != 2 {
		t.Fatalf("delay dims = %dx%d, want 2x2", r, c)
	}
	if result.Delay.At(0, 0) != 0.5 {
		t.Errorf("delay[0][0] = %v, want 0.5", result.Delay.At(0, 0))
	}
	if result.Delay.At(1, 1) != 1.0 {
		t.Errorf("delay[1][1] = %v, want 1.0", result.Delay.At(1, 1))
	}
	if result.Delay.At(0, 1) != 0 || result.Delay.At(1, 0) != 0 {
		t.Error("off-diagonal delays should be 0")
	}
}

func TestSafeFeedback_DiscreteInputDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = plant.SetInputDelay([]float64{3})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.8}), 0.1)

	cl, err := SafeFeedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Error("closed-loop should have no delay")
	}

	plantAbs, _ := plant.AbsorbDelay()
	clManual, _ := Feedback(plantAbs, controller, -1)

	nCl, _, _ := cl.Dims()
	nManual, _, _ := clManual.Dims()
	if nCl != nManual {
		t.Errorf("state count %d != manual %d", nCl, nManual)
	}

	steps := 15
	u := mat.NewDense(1, steps, nil)
	for k := 0; k < steps; k++ {
		u.Set(0, k, 1)
	}
	clResp, _ := cl.Simulate(u, nil, nil)
	manualResp, _ := clManual.Simulate(u, nil, nil)
	if !matEqual(clResp.Y, manualResp.Y, 1e-10) {
		t.Error("SafeFeedback != manual absorb+feedback")
	}
}

func TestSafeFeedback_DiscreteOutputDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = plant.SetOutputDelay([]float64{2})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0.1)

	cl, err := SafeFeedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Error("closed-loop should have no delay")
	}
}

func TestSafeFeedback_DiscreteIODelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(2, 2, []float64{0.8, 0.1, 0, 0.9}),
		mat.NewDense(2, 1, []float64{1, 0.5}),
		mat.NewDense(1, 2, []float64{1, 0.3}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	plant.Delay = mat.NewDense(1, 1, []float64{4})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.3}), 0.1)

	cl, err := SafeFeedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Error("closed-loop should have no delay")
	}
}

func TestSafeFeedback_DiscreteControllerDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	controller, _ := New(
		mat.NewDense(1, 1, []float64{0.3}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = controller.SetInputDelay([]float64{2})

	cl, err := SafeFeedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Error("closed-loop should have no delay")
	}
}

func TestSafeFeedback_NoDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.8}), 0.1)

	cl, err := SafeFeedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}

	clDirect, _ := Feedback(plant, controller, -1)
	nCl, _, _ := cl.Dims()
	nDirect, _, _ := clDirect.Dims()
	if nCl != nDirect {
		t.Errorf("state count %d != direct %d", nCl, nDirect)
	}
}

func TestSafeFeedback_ContinuousPade(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	plant.Delay = mat.NewDense(1, 1, []float64{0.3})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.8}), 0)

	cl, err := SafeFeedback(plant, controller, -1, WithPadeOrder(3))
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Error("closed-loop should have no delay")
	}
	if !cl.IsContinuous() {
		t.Error("closed-loop should be continuous")
	}

	padeApprox, _ := PadeDelay(0.3, 3)
	plantNoDelay := plant.Copy()
	plantNoDelay.Delay = nil
	plantApprox, _ := Series(padeApprox, plantNoDelay)
	clManual, _ := Feedback(plantApprox, controller, -1)

	nCl, _, _ := cl.Dims()
	nManual, _, _ := clManual.Dims()
	if nCl != nManual {
		t.Errorf("state count %d != manual %d", nCl, nManual)
	}

	stable, _ := cl.IsStable()
	if !stable {
		t.Error("closed-loop should be stable")
	}
}

func TestSafeFeedback_ContinuousInputDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = plant.SetInputDelay([]float64{0.5})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)

	cl, err := SafeFeedback(plant, controller, -1, WithPadeOrder(5))
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Error("should have no delay")
	}
	nCl, _, _ := cl.Dims()
	if nCl != 1+5 {
		t.Errorf("n = %d, want 6 (1 plant + 5 Pade)", nCl)
	}
}

func TestSafeFeedback_ContinuousOutputDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = plant.SetOutputDelay([]float64{0.4})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)

	cl, err := SafeFeedback(plant, controller, -1, WithPadeOrder(3))
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Error("should have no delay")
	}
	nCl, _, _ := cl.Dims()
	if nCl != 1+3 {
		t.Errorf("n = %d, want 4 (1 plant + 3 Pade)", nCl)
	}
}

func TestSafeFeedback_SingularAlgebraicLoop(t *testing.T) {
	plant, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0.1)
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0.1)

	_, err := SafeFeedback(plant, controller, 1)
	if !errors.Is(err, ErrSingularTransform) {
		t.Errorf("expected ErrSingularTransform, got %v", err)
	}
}

func TestSafeFeedback_DomainMismatch(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0.1)

	_, err := SafeFeedback(plant, controller, -1)
	if !errors.Is(err, ErrDomainMismatch) {
		t.Errorf("expected ErrDomainMismatch, got %v", err)
	}
}

func TestSafeFeedback_PositiveFeedback(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = plant.SetInputDelay([]float64{2})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.3}), 0.1)

	cl, err := SafeFeedback(plant, controller, 1)
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Error("should have no delay")
	}
}

func TestSafeFeedback_WellPosednessCheck(t *testing.T) {
	// Odd-order Pade has D=(-1)^N=-1. With negative feedback and unit gains:
	// I - (-1)*(-1)*1 = I - 1 = 0 → singular.
	plant, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	plant.InputDelay = []float64{0.5}
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)

	_, err := SafeFeedback(plant, controller, -1, WithPadeOrder(3))
	if err == nil {
		t.Fatal("expected singular algebraic loop error for odd-order Pade with unit gains and negative feedback")
	}

	// Even-order Pade has D=1: I - (-1)*1*1 = 2 → not singular.
	cl, err := SafeFeedback(plant, controller, -1, WithPadeOrder(2))
	if err != nil {
		t.Fatalf("even-order Pade should work: %v", err)
	}
	if cl.HasDelay() {
		t.Error("should have no delay")
	}
}

func TestSafeFeedback_DefaultPadeOrder(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	plant.Delay = mat.NewDense(1, 1, []float64{0.2})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0)

	cl, err := SafeFeedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	nCl, _, _ := cl.Dims()
	if nCl != 1+5 {
		t.Errorf("n = %d, want 6 (1 plant + 5 default Pade order)", nCl)
	}
}

func TestSafeFeedback_ContinuousMIMO(t *testing.T) {
	plant, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, nil),
		0,
	)
	_ = plant.SetInputDelay([]float64{0.2, 0.5})
	controller, _ := New(
		mat.NewDense(1, 1, []float64{-3}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(2, 1, []float64{0.5, 0.3}),
		mat.NewDense(2, 2, nil),
		0,
	)

	cl, err := SafeFeedback(plant, controller, -1, WithPadeOrder(3))
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Error("should have no delay")
	}
	nCl, _, _ := cl.Dims()
	if nCl != 2+3+3+1 {
		t.Errorf("n = %d, want 9 (2 plant + 3+3 Pade + 1 controller)", nCl)
	}
}

func TestSeriesLFT_IncompatibleIODelay(t *testing.T) {
	sys1, _ := NewFromSlices(1, 2, 2,
		[]float64{0.5}, []float64{1, 0}, []float64{1, 1}, []float64{0, 0, 0, 0}, 1.0)
	sys1.Delay = mat.NewDense(2, 2, []float64{2, 3, 5, 3})

	sys2, _ := NewFromSlices(1, 2, 1,
		[]float64{0.8}, []float64{1, 1}, []float64{1}, []float64{0, 0}, 1.0)
	sys2.Delay = mat.NewDense(1, 2, []float64{1, 1})

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if !result.HasInternalDelay() {
		t.Fatal("incompatible IO delays should produce InternalDelay")
	}
	n, m, p := result.Dims()
	if m != 2 || p != 1 {
		t.Errorf("dims m=%d p=%d, want m=2 p=1", m, p)
	}
	_ = n
}

func TestSeriesLFT_WithExistingInternalDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys1.SetInternalDelay(
		[]float64{0.5},
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{0.2}),
		mat.NewDense(1, 1, []float64{0.3}),
		mat.NewDense(1, 1, []float64{0.4}),
		mat.NewDense(1, 1, []float64{0}),
	)

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if !result.HasInternalDelay() {
		t.Fatal("should preserve InternalDelay from sys1")
	}
	if len(result.InternalDelay) != 1 {
		t.Errorf("InternalDelay count = %d, want 1", len(result.InternalDelay))
	}
	if result.InternalDelay[0] != 0.5 {
		t.Errorf("InternalDelay[0] = %v, want 0.5", result.InternalDelay[0])
	}
	n, m, p := result.Dims()
	if n != 2 || m != 1 || p != 1 {
		t.Errorf("dims = (%d,%d,%d), want (2,1,1)", n, m, p)
	}
}

func TestSeriesLFT_BothWithInternalDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys1.SetInternalDelay(
		[]float64{0.5},
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{0.2}),
		mat.NewDense(1, 1, []float64{0.3}),
		mat.NewDense(1, 1, []float64{0.4}),
		mat.NewDense(1, 1, []float64{0}),
	)

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys2.SetInternalDelay(
		[]float64{1.5, 2.0},
		mat.NewDense(1, 2, []float64{0.1, 0.2}),
		mat.NewDense(2, 1, []float64{0.3, 0.4}),
		mat.NewDense(1, 2, []float64{0.5, 0.6}),
		mat.NewDense(2, 1, []float64{0.7, 0.8}),
		mat.NewDense(2, 2, nil),
	)

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.InternalDelay) != 3 {
		t.Errorf("InternalDelay count = %d, want 3 (1+2)", len(result.InternalDelay))
	}
	if result.InternalDelay[0] != 0.5 {
		t.Errorf("InternalDelay[0] = %v, want 0.5", result.InternalDelay[0])
	}
	if result.InternalDelay[1] != 1.5 {
		t.Errorf("InternalDelay[1] = %v, want 1.5", result.InternalDelay[1])
	}
	if result.InternalDelay[2] != 2.0 {
		t.Errorf("InternalDelay[2] = %v, want 2.0", result.InternalDelay[2])
	}
}

func TestParallelLFT_MismatchedIODelay(t *testing.T) {
	sys1, _ := NewFromSlices(1, 1, 1,
		[]float64{0.5}, []float64{1}, []float64{1}, []float64{0}, 1.0)
	sys1.Delay = mat.NewDense(1, 1, []float64{2})

	sys2, _ := NewFromSlices(1, 1, 1,
		[]float64{0.8}, []float64{1}, []float64{1}, []float64{0}, 1.0)
	sys2.Delay = mat.NewDense(1, 1, []float64{5})

	result, err := Parallel(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if !result.HasInternalDelay() {
		t.Fatal("mismatched IO delays should produce InternalDelay")
	}
	if len(result.InternalDelay) != 2 {
		t.Errorf("InternalDelay count = %d, want 2", len(result.InternalDelay))
	}
	_, m, p := result.Dims()
	if m != 1 || p != 1 {
		t.Errorf("dims m=%d p=%d, want m=1 p=1", m, p)
	}
}

func TestParallelLFT_WithExistingInternalDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys1.SetInternalDelay(
		[]float64{0.3},
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{0.2}),
		mat.NewDense(1, 1, []float64{0.3}),
		mat.NewDense(1, 1, []float64{0.4}),
		mat.NewDense(1, 1, []float64{0}),
	)

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys2.SetInternalDelay(
		[]float64{0.7},
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{0.6}),
		mat.NewDense(1, 1, []float64{0.7}),
		mat.NewDense(1, 1, []float64{0.8}),
		mat.NewDense(1, 1, []float64{0}),
	)

	result, err := Parallel(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.InternalDelay) != 2 {
		t.Errorf("InternalDelay count = %d, want 2 (1+1)", len(result.InternalDelay))
	}
	if result.InternalDelay[0] != 0.3 {
		t.Errorf("InternalDelay[0] = %v, want 0.3", result.InternalDelay[0])
	}
	if result.InternalDelay[1] != 0.7 {
		t.Errorf("InternalDelay[1] = %v, want 0.7", result.InternalDelay[1])
	}
}

func TestAppend_WithInternalDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys1.SetInternalDelay(
		[]float64{0.5},
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{0.2}),
		mat.NewDense(1, 1, []float64{0.3}),
		mat.NewDense(1, 1, []float64{0.4}),
		mat.NewDense(1, 1, []float64{0}),
	)

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys2.SetInternalDelay(
		[]float64{1.0, 2.0},
		mat.NewDense(1, 2, []float64{0.5, 0.6}),
		mat.NewDense(2, 1, []float64{0.7, 0.8}),
		mat.NewDense(1, 2, []float64{0.9, 1.0}),
		mat.NewDense(2, 1, []float64{1.1, 1.2}),
		mat.NewDense(2, 2, nil),
	)

	result, err := Append(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.InternalDelay) != 3 {
		t.Fatalf("InternalDelay count = %d, want 3 (1+2)", len(result.InternalDelay))
	}
	if result.InternalDelay[0] != 0.5 {
		t.Errorf("InternalDelay[0] = %v, want 0.5", result.InternalDelay[0])
	}
	if result.InternalDelay[1] != 1.0 {
		t.Errorf("InternalDelay[1] = %v, want 1.0", result.InternalDelay[1])
	}
	if result.InternalDelay[2] != 2.0 {
		t.Errorf("InternalDelay[2] = %v, want 2.0", result.InternalDelay[2])
	}

	n, m, p := result.Dims()
	if n != 2 || m != 2 || p != 2 {
		t.Errorf("dims = (%d,%d,%d), want (2,2,2)", n, m, p)
	}

	r, c := result.B2.Dims()
	if r != 2 || c != 3 {
		t.Errorf("B2 dims = %dx%d, want 2x3", r, c)
	}
	if result.B2.At(0, 0) != 0.1 {
		t.Errorf("B2[0,0] = %v, want 0.1", result.B2.At(0, 0))
	}
	if result.B2.At(1, 1) != 0.5 {
		t.Errorf("B2[1,1] = %v, want 0.5", result.B2.At(1, 1))
	}
	if result.B2.At(0, 1) != 0 {
		t.Errorf("B2[0,1] = %v, want 0 (block-diagonal)", result.B2.At(0, 1))
	}

	r, c = result.D22.Dims()
	if r != 3 || c != 3 {
		t.Errorf("D22 dims = %dx%d, want 3x3", r, c)
	}
}

func TestSeriesLFT_Roundtrip(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys1.SetInternalDelay(
		[]float64{0.5},
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{0.2}),
		mat.NewDense(1, 1, []float64{0.3}),
		mat.NewDense(1, 1, []float64{0.4}),
		mat.NewDense(1, 1, []float64{0}),
	)

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys2.SetInternalDelay(
		[]float64{1.5},
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{0.6}),
		mat.NewDense(1, 1, []float64{0.7}),
		mat.NewDense(1, 1, []float64{0.8}),
		mat.NewDense(1, 1, []float64{0}),
	)

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}

	H, tau := result.GetDelayModel()
	if len(tau) != 2 {
		t.Fatalf("tau length = %d, want 2", len(tau))
	}

	rebuilt, err := SetDelayModel(H, tau)
	if err != nil {
		t.Fatal(err)
	}
	if len(rebuilt.InternalDelay) != 2 {
		t.Fatalf("rebuilt InternalDelay = %d, want 2", len(rebuilt.InternalDelay))
	}
	for i, v := range result.InternalDelay {
		if math.Abs(rebuilt.InternalDelay[i]-v) > 1e-12 {
			t.Errorf("InternalDelay[%d] = %v, want %v", i, rebuilt.InternalDelay[i], v)
		}
	}
}

func TestSeriesLFT_IncompatibleMIMOIODelay(t *testing.T) {
	sys1, _ := NewFromSlices(1, 1, 2,
		[]float64{0.5}, []float64{1}, []float64{1, 1}, []float64{0, 0}, 0)
	sys1.Delay = mat.NewDense(2, 1, []float64{1, 3})

	sys2, _ := NewFromSlices(1, 2, 1,
		[]float64{0.8}, []float64{1, 1}, []float64{1}, []float64{0, 0}, 0)
	sys2.Delay = mat.NewDense(1, 2, []float64{2, 4})

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}

	if !result.HasInternalDelay() {
		t.Fatal("incompatible MIMO IO delays should produce InternalDelay")
	}

	_, m, p := result.Dims()
	if m != 1 || p != 1 {
		t.Errorf("dims m=%d p=%d, want m=1 p=1", m, p)
	}

	H, tau := result.GetDelayModel()
	if len(tau) == 0 {
		t.Fatal("should have non-empty tau")
	}
	_, mH, pH := H.Dims()
	if mH != m+len(tau) {
		t.Errorf("H inputs = %d, want %d", mH, m+len(tau))
	}
	if pH != p+len(tau) {
		t.Errorf("H outputs = %d, want %d", pH, p+len(tau))
	}
}

func TestParallelLFT_MatchingDelaysNoLFT(t *testing.T) {
	sys1, _ := NewFromSlices(1, 1, 1,
		[]float64{0.5}, []float64{1}, []float64{1}, []float64{0}, 1.0)
	sys1.Delay = mat.NewDense(1, 1, []float64{3})

	sys2, _ := NewFromSlices(1, 1, 1,
		[]float64{0.8}, []float64{1}, []float64{1}, []float64{0}, 1.0)
	sys2.Delay = mat.NewDense(1, 1, []float64{3})

	result, err := Parallel(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if result.HasInternalDelay() {
		t.Error("matching delays should not produce InternalDelay")
	}
	if result.Delay == nil || result.Delay.At(0, 0) != 3 {
		t.Errorf("matching delays should preserve IODelay=3")
	}
}

func TestAppend_MixedInternalAndIO(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys1.SetInternalDelay(
		[]float64{0.5},
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{0.2}),
		mat.NewDense(1, 1, []float64{0.3}),
		mat.NewDense(1, 1, []float64{0.4}),
		mat.NewDense(1, 1, []float64{0}),
	)

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	result, err := Append(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.InternalDelay) != 1 {
		t.Errorf("InternalDelay count = %d, want 1", len(result.InternalDelay))
	}
	r, c := result.B2.Dims()
	if r != 2 || c != 1 {
		t.Errorf("B2 dims = %dx%d, want 2x1", r, c)
	}
	if result.B2.At(1, 0) != 0 {
		t.Errorf("B2[1,0] = %v, want 0 (block-diagonal)", result.B2.At(1, 0))
	}
}

func TestFeedback_LFT_InputDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = plant.SetInputDelay([]float64{3})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.8}), 0.1)

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasInternalDelay() {
		t.Error("plant InputDelay should stay external, not become internal")
	}
	if cl.InputDelay == nil || len(cl.InputDelay) != 1 || cl.InputDelay[0] != 3 {
		t.Errorf("InputDelay = %v, want [3]", cl.InputDelay)
	}
	n, m, p := cl.Dims()
	if m != 1 || p != 1 {
		t.Errorf("dims m=%d p=%d, want 1,1", m, p)
	}
	_ = n
}

func TestFeedback_LFT_OutputDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = plant.SetOutputDelay([]float64{2})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0.1)

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !cl.HasInternalDelay() {
		t.Error("should have internal delays")
	}
	if len(cl.InternalDelay) != 1 || cl.InternalDelay[0] != 2 {
		t.Errorf("InternalDelay = %v, want [2]", cl.InternalDelay)
	}
}

func TestFeedback_LFT_BothDelayed(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = plant.SetInputDelay([]float64{2})
	controller, _ := New(
		mat.NewDense(1, 1, []float64{0.3}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = controller.SetOutputDelay([]float64{1})

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !cl.HasInternalDelay() {
		t.Error("controller OutputDelay should become internal")
	}
	if len(cl.InternalDelay) != 1 {
		t.Errorf("expected 1 internal delay (controller OutputDelay), got %d", len(cl.InternalDelay))
	}
	if cl.InputDelay == nil || cl.InputDelay[0] != 2 {
		t.Errorf("plant InputDelay should stay external: got %v, want [2]", cl.InputDelay)
	}
}

func TestFeedback_LFT_NoDelay_Unchanged(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasInternalDelay() {
		t.Error("delay-free feedback should not have internal delays")
	}
	poles, _ := cl.Poles()
	if len(poles) != 1 || math.Abs(real(poles[0])+1) > 1e-10 {
		t.Errorf("pole = %v, want -1", poles[0])
	}
}

func TestFeedback_LFT_IODelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	plant.Delay = mat.NewDense(1, 1, []float64{0.3})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0)

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !cl.HasInternalDelay() {
		t.Error("should have internal delays from IODelay")
	}
	if len(cl.InternalDelay) != 1 || cl.InternalDelay[0] != 0.3 {
		t.Errorf("InternalDelay = %v, want [0.3]", cl.InternalDelay)
	}
	_, m, p := cl.Dims()
	if m != 1 || p != 1 {
		t.Errorf("dims m=%d p=%d, want 1,1", m, p)
	}
}

func TestFeedback_LFT_PositiveFeedback(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = plant.SetInputDelay([]float64{2})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.3}), 0.1)

	cl, err := Feedback(plant, controller, 1)
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasInternalDelay() {
		t.Error("plant InputDelay should stay external, not become internal")
	}
	if cl.InputDelay == nil || cl.InputDelay[0] != 2 {
		t.Errorf("plant InputDelay should stay external: got %v, want [2]", cl.InputDelay)
	}
}

func TestFeedbackKeepsInputDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = plant.SetInputDelay([]float64{5})
	_ = plant.SetOutputDelay([]float64{2})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0.1)

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if cl.InputDelay == nil || cl.InputDelay[0] != 5 {
		t.Errorf("plant InputDelay should stay external: got %v, want [5]", cl.InputDelay)
	}
	if !cl.HasInternalDelay() {
		t.Fatal("plant OutputDelay should become internal delay")
	}
	found := false
	for _, tau := range cl.InternalDelay {
		if tau == 2 {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("InternalDelay should contain 2 (from OutputDelay), got %v", cl.InternalDelay)
	}
}

func TestSeriesKeepsExternalDelays(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys1.InputDelay = []float64{1.5}
	sys1.OutputDelay = []float64{0.5}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2.InputDelay = []float64{0.3}
	sys2.OutputDelay = []float64{0.7}

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if result.InputDelay == nil || math.Abs(result.InputDelay[0]-1.5) > 1e-12 {
		t.Errorf("sys1 InputDelay should stay external: got %v, want [1.5]", result.InputDelay)
	}
	if result.OutputDelay == nil || math.Abs(result.OutputDelay[0]-0.7) > 1e-12 {
		t.Errorf("sys2 OutputDelay should stay external: got %v, want [0.7]", result.OutputDelay)
	}
}

func TestFeedback_DelayInFeedbackPath(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{-1.0 / 3.0}),
		mat.NewDense(1, 1, []float64{1.5 / 3.0}),
		mat.NewDense(1, 1, []float64{1.0}),
		mat.NewDense(1, 1, []float64{0.0}),
		0,
	)
	_ = plant.SetInputDelay([]float64{0.5})

	ctrl, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.25}),
		mat.NewDense(1, 1, []float64{0.5}),
		0,
	)

	L, _ := Series(ctrl, plant)

	I, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	S, _ := Feedback(I, L, -1)
	T, _ := Feedback(L, nil, -1)

	dt := 0.05
	nSteps := 1001
	u := mat.NewDense(1, nSteps, nil)
	for j := range nSteps {
		u.Set(0, j, 1.0)
	}

	Sd, _ := S.DiscretizeZOH(dt)
	sResp, _ := Sd.Simulate(u, nil, nil)
	sFinal := sResp.Y.At(0, nSteps-1)

	Td, _ := T.DiscretizeZOH(dt)
	tResp, _ := Td.Simulate(u, nil, nil)
	tFinal := tResp.Y.At(0, nSteps-1)

	if math.Abs(sFinal) > 0.05 {
		t.Errorf("S(inf) = %.4f, want ~0", sFinal)
	}
	if math.Abs(tFinal-1.0) > 0.05 {
		t.Errorf("T(inf) = %.4f, want ~1", tFinal)
	}
}
