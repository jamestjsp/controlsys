package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func makeTestSystem() *System {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)
	return sys
}

func TestDiscretize_WrongDomain(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_, err := sys.Discretize(0.1)
	if err == nil {
		t.Fatal("expected ErrWrongDomain")
	}
}

func TestDiscretize_Roundtrip(t *testing.T) {
	orig := makeTestSystem()
	dt := 0.01

	disc, err := orig.Discretize(dt)
	if err != nil {
		t.Fatal(err)
	}
	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}
	if disc.Dt != dt {
		t.Fatalf("Dt = %v, want %v", disc.Dt, dt)
	}

	rec, err := disc.Undiscretize()
	if err != nil {
		t.Fatal(err)
	}
	if !rec.IsContinuous() {
		t.Fatal("expected continuous")
	}

	tol := 1e-10
	assertMatClose(t, "A", rec.A, orig.A, tol)
	assertMatClose(t, "B", rec.B, orig.B, tol)
	assertMatClose(t, "C", rec.C, orig.C, tol)
	assertMatClose(t, "D", rec.D, orig.D, tol)
}

func TestDiscretize_EigenvalueMapping(t *testing.T) {
	sys := makeTestSystem()
	contPoles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range contPoles {
		if real(p) >= 0 {
			t.Fatalf("continuous pole %v has non-negative real part", p)
		}
	}

	disc, err := sys.Discretize(0.01)
	if err != nil {
		t.Fatal(err)
	}
	discPoles, err := disc.Poles()
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range discPoles {
		if cmplx.Abs(p) >= 1 {
			t.Errorf("discrete pole %v has |z| >= 1", p)
		}
	}
}

func TestDiscretize_SingularTransform(t *testing.T) {
	dt := 1.0
	beta := 2.0 / dt
	A := mat.NewDense(2, 2, []float64{beta, 0, 0, -1})
	B := mat.NewDense(2, 1, []float64{1, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)

	_, err := sys.Discretize(dt)
	if err == nil {
		t.Fatal("expected ErrSingularTransform")
	}
}

func TestDiscretize_N0_GainOnly(t *testing.T) {
	D := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	sys := &System{
		A:  &mat.Dense{},
		B:  &mat.Dense{},
		C:  &mat.Dense{},
		D:  D,
		Dt: 0,
	}

	disc, err := sys.Discretize(0.1)
	if err != nil {
		t.Fatal(err)
	}
	assertMatClose(t, "D", disc.D, D, 1e-15)
	n, _, _ := disc.Dims()
	if n != 0 {
		t.Fatalf("expected n=0, got %d", n)
	}
}

func TestDiscretize_N_GT0_M0(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0, 0, -2})
	sys, _ := New(A, nil, nil, nil, 0)

	disc, err := sys.Discretize(0.1)
	if err != nil {
		t.Fatal(err)
	}
	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}
	n, m, _ := disc.Dims()
	if n != 2 || m != 0 {
		t.Fatalf("expected n=2 m=0, got n=%d m=%d", n, m)
	}

	rec, err := disc.Undiscretize()
	if err != nil {
		t.Fatal(err)
	}
	assertMatClose(t, "A", rec.A, A, 1e-10)
}

func TestUndiscretize_WrongDomain(t *testing.T) {
	sys := makeTestSystem()
	_, err := sys.Undiscretize()
	if err == nil {
		t.Fatal("expected ErrWrongDomain")
	}
}

func TestUndiscretize_SingularTransform(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0, 0, 0})
	B := mat.NewDense(2, 1, []float64{1, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0.1)

	_, err := sys.Undiscretize()
	if err == nil {
		t.Fatal("expected ErrSingularTransform for eigenvalue at -1")
	}
}

func TestZOH_WrongDomain(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_, err := sys.DiscretizeZOH(0.1)
	if err == nil {
		t.Fatal("expected ErrWrongDomain")
	}
}

func TestZOH_N0_GainOnly(t *testing.T) {
	D := mat.NewDense(1, 1, []float64{5})
	sys := &System{
		A:  &mat.Dense{},
		B:  &mat.Dense{},
		C:  &mat.Dense{},
		D:  D,
		Dt: 0,
	}

	disc, err := sys.DiscretizeZOH(0.1)
	if err != nil {
		t.Fatal(err)
	}
	assertMatClose(t, "D", disc.D, D, 1e-15)
}

func TestZOH_ScalarAnalytical(t *testing.T) {
	a := -2.0
	b := 3.0
	dt := 0.1

	A := mat.NewDense(1, 1, []float64{a})
	B := mat.NewDense(1, 1, []float64{b})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)

	disc, err := sys.DiscretizeZOH(dt)
	if err != nil {
		t.Fatal(err)
	}

	// Analytical: Ad = exp(a*dt), Bd = (exp(a*dt)-1)/a * b
	adExpected := math.Exp(a * dt)
	bdExpected := (math.Exp(a*dt) - 1) / a * b

	assertMatClose(t, "Ad", disc.A, mat.NewDense(1, 1, []float64{adExpected}), 1e-12)
	assertMatClose(t, "Bd", disc.B, mat.NewDense(1, 1, []float64{bdExpected}), 1e-12)
	assertMatClose(t, "C", disc.C, C, 1e-15)
	assertMatClose(t, "D", disc.D, D, 1e-15)
}

func TestZOH_StabilityPreserved(t *testing.T) {
	sys := makeTestSystem()
	stable, err := sys.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Fatal("continuous system should be stable")
	}

	disc, err := sys.DiscretizeZOH(0.01)
	if err != nil {
		t.Fatal(err)
	}
	dStable, err := disc.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !dStable {
		t.Fatal("ZOH-discretized system should be stable")
	}

	dPoles, err := disc.Poles()
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range dPoles {
		if cmplx.Abs(p) >= 1 {
			t.Errorf("discrete pole %v has |z| >= 1", p)
		}
	}
}

func TestZOH_2x2(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)

	dt := 0.1
	disc, err := sys.DiscretizeZOH(dt)
	if err != nil {
		t.Fatal(err)
	}

	// For double integrator: expm([0 1; 0 0]*dt) = [1 dt; 0 1]
	// Full block: expm([0 1 0; 0 0 1; 0 0 0]*dt) = [1 dt dt^2/2; 0 1 dt; 0 0 1]
	// So Ad = [1 dt; 0 1], Bd = [dt^2/2; dt]
	AdExpected := mat.NewDense(2, 2, []float64{1, dt, 0, 1})
	BdExpected := mat.NewDense(2, 1, []float64{dt * dt / 2, dt})

	assertMatClose(t, "Ad", disc.A, AdExpected, 1e-12)
	assertMatClose(t, "Bd", disc.B, BdExpected, 1e-12)
}

func TestDiscretizeWithOpts_ZOH_IntegerInputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.3}

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh"})
	if err != nil {
		t.Fatal(err)
	}
	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}
	if len(disc.InputDelay) != 1 || disc.InputDelay[0] != 3 {
		t.Errorf("InputDelay = %v, want [3]", disc.InputDelay)
	}
}

func TestDiscretizeWithOpts_Tustin_IntegerOutputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.OutputDelay = []float64{0.5}

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "tustin"})
	if err != nil {
		t.Fatal(err)
	}
	if len(disc.OutputDelay) != 1 || disc.OutputDelay[0] != 5 {
		t.Errorf("OutputDelay = %v, want [5]", disc.OutputDelay)
	}
}

func TestDiscretizeWithOpts_FractionalInputDelay_NoThiran_Error(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.35}

	_, err := sys.DiscretizeWithOpts(0.1, C2DOptions{Method: "zoh"})
	if err == nil {
		t.Fatal("expected error for fractional delay without Thiran")
	}
}

func TestDiscretizeWithOpts_FractionalInputDelay_WithThiran(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.35}

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh", ThiranOrder: 3})
	if err != nil {
		t.Fatal(err)
	}
	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}
	if disc.InputDelay != nil {
		t.Errorf("InputDelay should be nil after Thiran absorption, got %v", disc.InputDelay)
	}
	n, _, _ := disc.Dims()
	if n <= 1 {
		t.Errorf("state dim = %d, expected > 1 (plant + Thiran)", n)
	}
}

func TestDiscretizeWithOpts_FractionalOutputDelay_WithThiran(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.OutputDelay = []float64{0.35}

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh", ThiranOrder: 3})
	if err != nil {
		t.Fatal(err)
	}
	if disc.OutputDelay != nil {
		t.Errorf("OutputDelay should be nil after Thiran absorption, got %v", disc.OutputDelay)
	}
	n, _, _ := disc.Dims()
	if n <= 1 {
		t.Errorf("state dim = %d, expected > 1 (plant + Thiran)", n)
	}
}

func TestDiscretizeWithOpts_MIMO_FractionalInputDelay_WithThiran(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	sys.InputDelay = []float64{0.35, 0.2}

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh", ThiranOrder: 3})
	if err != nil {
		t.Fatal(err)
	}
	if disc.InputDelay != nil {
		t.Errorf("InputDelay should be nil after Thiran absorption, got %v", disc.InputDelay)
	}
	_, m, p := disc.Dims()
	if m != 2 || p != 2 {
		t.Errorf("dims m=%d p=%d, want 2,2", m, p)
	}
}

func TestDiscretizeWithOpts_MixedIntegerFractionalInputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{0, 0}),
		0,
	)
	sys.InputDelay = []float64{0.3, 0.35}

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh", ThiranOrder: 3})
	if err != nil {
		t.Fatal(err)
	}
	_, m, _ := disc.Dims()
	if m != 2 {
		t.Errorf("m = %d, want 2", m)
	}
}

func TestDiscretizeWithOpts_DefaultMethod(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	disc, err := sys.DiscretizeWithOpts(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}
}

func TestDiscretizeWithOpts_AlreadyDiscrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_, err := sys.DiscretizeWithOpts(0.1, C2DOptions{})
	if err == nil {
		t.Fatal("expected error for already discrete")
	}
}

func TestDiscretizeWithOpts_FreqResponseWithThiran(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.35}

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh", ThiranOrder: 3})
	if err != nil {
		t.Fatal(err)
	}

	tfDisc, _ := disc.TransferFunction(nil)

	for _, w := range []float64{0.1, 0.5, 1.0} {
		z := cmplx.Exp(complex(0, w*dt))
		h := tfDisc.TF.Eval(z)[0][0]
		mag := cmplx.Abs(h)
		if mag > 2.0 || mag < 0 {
			t.Errorf("w=%v: |H(z)| = %v out of reasonable range", w, mag)
		}
	}
}

func TestDiscretizeWithOpts_ZeroDelay_NoThiranNeeded(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0}

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh"})
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := disc.Dims()
	if n != 1 {
		t.Errorf("state dim = %d, want 1 (no Thiran for zero delay)", n)
	}
}

func TestDiscretize_DoesNotMutateReceiver(t *testing.T) {
	orig := makeTestSystem()
	before := orig.Copy()

	_, err := orig.Discretize(0.01)
	if err != nil {
		t.Fatal(err)
	}

	assertMatClose(t, "A", orig.A, before.A, 0)
	assertMatClose(t, "B", orig.B, before.B, 0)
	assertMatClose(t, "C", orig.C, before.C, 0)
	assertMatClose(t, "D", orig.D, before.D, 0)
}

func TestDiscretizeWithOpts_SISO_FractionalIODelay_Thiran(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.Delay = mat.NewDense(1, 1, []float64{0.35})

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh", ThiranOrder: 3})
	if err != nil {
		t.Fatal(err)
	}

	if disc.Delay != nil {
		t.Error("expected Delay=nil after Thiran decomposition")
	}

	tfDisc, _ := disc.TransferFunction(nil)
	for _, w := range []float64{0.1, 0.5, 1.0} {
		z := cmplx.Exp(complex(0, w*dt))
		h := tfDisc.TF.Eval(z)[0][0]
		mag := cmplx.Abs(h)
		if mag > 2.0 || mag < 0 {
			t.Errorf("w=%v: |H(z)| = %v out of reasonable range", w, mag)
		}
	}
}

func TestDiscretizeWithOpts_MIMO_DecomposableIODelay_Thiran(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 2, []float64{1, 0.5}),
		mat.NewDense(2, 1, []float64{1, 0.3}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	sys.Delay = mat.NewDense(2, 2, []float64{0.3, 0.5, 0.3, 0.5})

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh", ThiranOrder: 3})
	if err != nil {
		t.Fatal(err)
	}

	if disc.Delay != nil {
		t.Error("expected Delay=nil for column-uniform decomposable IODelay")
	}
}

func TestDiscretizeWithOpts_SISO_IntegerIODelay_NoThiran(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.Delay = mat.NewDense(1, 1, []float64{0.3})

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh", ThiranOrder: 0})
	if err != nil {
		t.Fatal(err)
	}

	if disc.Delay == nil {
		t.Fatal("expected discrete Delay to be set")
	}
	if got := disc.Delay.At(0, 0); got != 3 {
		t.Errorf("Delay = %v, want 3", got)
	}
}

func TestDiscretizeWithOpts_MixedIODelay_InputDelay_Thiran(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.Delay = mat.NewDense(1, 1, []float64{0.25})
	sys.InputDelay = []float64{0.15}

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh", ThiranOrder: 3})
	if err != nil {
		t.Fatal(err)
	}

	if disc.Delay != nil {
		t.Error("expected Delay=nil after decomposition+merge")
	}

	tfDisc, _ := disc.TransferFunction(nil)
	z := cmplx.Exp(complex(0, 0.5*dt))
	h := tfDisc.TF.Eval(z)[0][0]
	mag := cmplx.Abs(h)
	if mag > 2.0 || mag < 0 {
		t.Errorf("w=0.5: |H(z)| = %v out of reasonable range", mag)
	}

	_ = math.Abs(0) // keep math import
}

func TestUndiscretizeInputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.9}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	sys.InputDelay = []float64{3}

	ct, err := sys.Undiscretize()
	if err != nil {
		t.Fatal(err)
	}
	if len(ct.InputDelay) != 1 || math.Abs(ct.InputDelay[0]-0.3) > 1e-12 {
		t.Errorf("InputDelay = %v, want [0.3]", ct.InputDelay)
	}
}

func TestUndiscretizeOutputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.9}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	sys.OutputDelay = []float64{5}

	ct, err := sys.Undiscretize()
	if err != nil {
		t.Fatal(err)
	}
	if len(ct.OutputDelay) != 1 || math.Abs(ct.OutputDelay[0]-0.5) > 1e-12 {
		t.Errorf("OutputDelay = %v, want [0.5]", ct.OutputDelay)
	}
}

func TestUndiscretizeInternalDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.9}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	sys.InternalDelay = []float64{5}
	sys.B2 = mat.NewDense(1, 1, []float64{0.2})
	sys.C2 = mat.NewDense(1, 1, []float64{0.3})
	sys.D12 = mat.NewDense(1, 1, []float64{0.4})
	sys.D21 = mat.NewDense(1, 1, []float64{0.5})
	sys.D22 = mat.NewDense(1, 1, []float64{0})

	ct, err := sys.Undiscretize()
	if err != nil {
		t.Fatal(err)
	}
	if len(ct.InternalDelay) != 1 || math.Abs(ct.InternalDelay[0]-0.5) > 1e-12 {
		t.Errorf("InternalDelay = %v, want [0.5]", ct.InternalDelay)
	}
	assertMatClose(t, "B2", ct.B2, sys.B2, 1e-15)
	assertMatClose(t, "C2", ct.C2, sys.C2, 1e-15)
	assertMatClose(t, "D12", ct.D12, sys.D12, 1e-15)
	assertMatClose(t, "D21", ct.D21, sys.D21, 1e-15)
	assertMatClose(t, "D22", ct.D22, sys.D22, 1e-15)
}

func TestUndiscretizeAllDelays(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.9}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	sys.Delay = mat.NewDense(1, 1, []float64{2})
	sys.InputDelay = []float64{3}
	sys.OutputDelay = []float64{4}

	ct, err := sys.Undiscretize()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(ct.Delay.At(0, 0)-0.2) > 1e-12 {
		t.Errorf("Delay = %v, want 0.2", ct.Delay.At(0, 0))
	}
	if math.Abs(ct.InputDelay[0]-0.3) > 1e-12 {
		t.Errorf("InputDelay = %v, want [0.3]", ct.InputDelay)
	}
	if math.Abs(ct.OutputDelay[0]-0.4) > 1e-12 {
		t.Errorf("OutputDelay = %v, want [0.4]", ct.OutputDelay)
	}
}

func TestRoundtripDelays(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.3}
	sys.OutputDelay = []float64{0.5}

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh"})
	if err != nil {
		t.Fatal(err)
	}
	if disc.InputDelay[0] != 3 {
		t.Fatalf("discrete InputDelay = %v, want 3", disc.InputDelay[0])
	}
	if disc.OutputDelay[0] != 5 {
		t.Fatalf("discrete OutputDelay = %v, want 5", disc.OutputDelay[0])
	}

	ct, err := disc.Undiscretize()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(ct.InputDelay[0]-0.3) > 1e-10 {
		t.Errorf("roundtrip InputDelay = %v, want 0.3", ct.InputDelay[0])
	}
	if math.Abs(ct.OutputDelay[0]-0.5) > 1e-10 {
		t.Errorf("roundtrip OutputDelay = %v, want 0.5", ct.OutputDelay[0])
	}
}

func TestDiscretizeZOHWithInputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.3}

	dt := 0.1
	disc, err := sys.DiscretizeZOH(dt)
	if err != nil {
		t.Fatal(err)
	}
	if len(disc.InputDelay) != 1 || disc.InputDelay[0] != 3 {
		t.Errorf("InputDelay = %v, want [3]", disc.InputDelay)
	}
}

func TestDiscretizeZOHWithOutputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.OutputDelay = []float64{0.5}

	dt := 0.1
	disc, err := sys.DiscretizeZOH(dt)
	if err != nil {
		t.Fatal(err)
	}
	if len(disc.OutputDelay) != 1 || disc.OutputDelay[0] != 5 {
		t.Errorf("OutputDelay = %v, want [5]", disc.OutputDelay)
	}
}

func TestDiscretizeTustinWithInputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.3}

	dt := 0.1
	disc, err := sys.Discretize(dt)
	if err != nil {
		t.Fatal(err)
	}
	if len(disc.InputDelay) != 1 || disc.InputDelay[0] != 3 {
		t.Errorf("InputDelay = %v, want [3]", disc.InputDelay)
	}
}

func TestDiscretizeTustinFractionalDelayError(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.35}

	_, err := sys.Discretize(0.1)
	if err == nil {
		t.Fatal("expected error for fractional delay")
	}
}

func TestDiscretizeZOHWithInternalDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InternalDelay = []float64{0.2}
	sys.B2 = mat.NewDense(2, 1, []float64{0.5, 0.3})
	sys.C2 = mat.NewDense(1, 2, []float64{0.1, 0.2})
	sys.D12 = mat.NewDense(1, 1, []float64{0.4})
	sys.D21 = mat.NewDense(1, 1, []float64{0.6})
	sys.D22 = mat.NewDense(1, 1, []float64{0})

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh"})
	if err != nil {
		t.Fatal(err)
	}
	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}
	if disc.Dt != dt {
		t.Fatalf("Dt = %v, want %v", disc.Dt, dt)
	}

	if len(disc.InternalDelay) != 1 || disc.InternalDelay[0] != 2 {
		t.Errorf("InternalDelay = %v, want [2]", disc.InternalDelay)
	}
	assertMatClose(t, "B2", disc.B2, sys.B2, 1e-15)
	assertMatClose(t, "C2", disc.C2, sys.C2, 1e-15)
	assertMatClose(t, "D12", disc.D12, sys.D12, 1e-15)
	assertMatClose(t, "D21", disc.D21, sys.D21, 1e-15)
	assertMatClose(t, "D22", disc.D22, sys.D22, 1e-15)

	refSys, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	ref, _ := refSys.DiscretizeZOH(dt)
	assertMatClose(t, "A", disc.A, ref.A, 1e-12)
	assertMatClose(t, "B", disc.B, ref.B, 1e-12)
	assertMatClose(t, "C", disc.C, ref.C, 1e-12)
	assertMatClose(t, "D", disc.D, ref.D, 1e-12)
}

func TestDiscretizeZOHInternalDelayD22UpperTriangular(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InternalDelay = []float64{0.2, 0.3}
	sys.B2 = mat.NewDense(1, 2, []float64{0.5, 0.3})
	sys.C2 = mat.NewDense(2, 1, []float64{0.1, 0.2})
	sys.D12 = mat.NewDense(1, 2, []float64{0.4, 0.1})
	sys.D21 = mat.NewDense(2, 1, []float64{0.6, 0.7})
	sys.D22 = mat.NewDense(2, 2, []float64{0, 0.5, 0, 0})

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh"})
	if err != nil {
		t.Fatal(err)
	}
	if len(disc.InternalDelay) != 2 {
		t.Fatalf("InternalDelay len = %d, want 2", len(disc.InternalDelay))
	}
	if disc.InternalDelay[0] != 2 || disc.InternalDelay[1] != 3 {
		t.Errorf("InternalDelay = %v, want [2 3]", disc.InternalDelay)
	}
}

func TestDiscretizeZOHInternalDelayD22General(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InternalDelay = []float64{0.2, 0.3}
	sys.B2 = mat.NewDense(1, 2, []float64{0.5, 0.3})
	sys.C2 = mat.NewDense(2, 1, []float64{0.1, 0.2})
	sys.D12 = mat.NewDense(1, 2, []float64{0.4, 0.1})
	sys.D21 = mat.NewDense(2, 1, []float64{0.6, 0.7})
	sys.D22 = mat.NewDense(2, 2, []float64{0.1, 0.5, 0.3, 0})

	_, err := sys.DiscretizeWithOpts(0.1, C2DOptions{Method: "zoh"})
	if err == nil {
		t.Fatal("expected error for non-upper-triangular D22")
	}
}

func TestDiscretizeZOHInternalDelayFreqResp(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.5}

	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}

	dt := 0.1
	disc, err := lft.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh"})
	if err != nil {
		t.Fatal(err)
	}

	if len(disc.InternalDelay) == 0 {
		t.Fatal("expected InternalDelay on discretized LFT system")
	}

	discTF, _ := disc.TransferFunction(nil)
	for _, w := range []float64{0.1, 0.5, 1.0, 2.0} {
		z := cmplx.Exp(complex(0, w*dt))
		h := discTF.TF.Eval(z)[0][0]
		mag := cmplx.Abs(h)
		if mag > 2.0 || mag < 0 {
			t.Errorf("w=%v: |H(z)| = %v out of range", w, mag)
		}
	}
}

func TestC2DDelayModelingInternal(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.35}

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh", DelayModeling: "internal"})
	if err != nil {
		t.Fatal(err)
	}
	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}

	n, _, _ := disc.Dims()
	if n != 1 {
		t.Errorf("state dim = %d, want 1 (no state augmentation)", n)
	}
	if len(disc.InternalDelay) != 1 {
		t.Fatalf("InternalDelay len = %d, want 1", len(disc.InternalDelay))
	}
	if math.Abs(disc.InternalDelay[0]-0.5) > 1e-9 {
		t.Errorf("InternalDelay[0] = %v, want 0.5", disc.InternalDelay[0])
	}
	if len(disc.InputDelay) != 1 || disc.InputDelay[0] != 3 {
		t.Errorf("InputDelay = %v, want [3]", disc.InputDelay)
	}
}

func TestC2DDelayModelingInternalFreqResp(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.35}
	dt := 0.1

	discInternal, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh", DelayModeling: "internal"})
	if err != nil {
		t.Fatal(err)
	}
	discState, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh", ThiranOrder: 3})
	if err != nil {
		t.Fatal(err)
	}

	for _, w := range []float64{0.1, 0.5, 1.0} {
		z := cmplx.Exp(complex(0, w*dt))

		hInt, err := discInternal.EvalFr(z)
		if err != nil {
			t.Fatalf("w=%v: EvalFr internal error: %v", w, err)
		}
		hState, err := discState.EvalFr(z)
		if err != nil {
			t.Fatalf("w=%v: EvalFr state error: %v", w, err)
		}

		magInt := cmplx.Abs(hInt[0][0])
		magState := cmplx.Abs(hState[0][0])
		if math.Abs(magInt-magState)/magState > 0.15 {
			t.Errorf("w=%v: mag mismatch internal=%v state=%v", w, magInt, magState)
		}
	}
}

func TestC2DDelayModelingDefault(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.3}

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh"})
	if err != nil {
		t.Fatal(err)
	}
	if len(disc.InputDelay) != 1 || disc.InputDelay[0] != 3 {
		t.Errorf("InputDelay = %v, want [3]", disc.InputDelay)
	}
	if disc.InternalDelay != nil {
		t.Errorf("InternalDelay should be nil for default (state) mode, got %v", disc.InternalDelay)
	}
}

func TestC2DDelayModelingIntegerDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.3}

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh", DelayModeling: "internal"})
	if err != nil {
		t.Fatal(err)
	}
	if len(disc.InputDelay) != 1 || disc.InputDelay[0] != 3 {
		t.Errorf("InputDelay = %v, want [3]", disc.InputDelay)
	}
	if disc.InternalDelay != nil {
		t.Errorf("InternalDelay should be nil for integer delay, got %v", disc.InternalDelay)
	}
}

func TestC2DDelayModelingInternalOutputDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.OutputDelay = []float64{0.35}

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh", DelayModeling: "internal"})
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := disc.Dims()
	if n != 1 {
		t.Errorf("state dim = %d, want 1 (no state augmentation)", n)
	}
	if len(disc.InternalDelay) != 1 {
		t.Fatalf("InternalDelay len = %d, want 1", len(disc.InternalDelay))
	}
	if math.Abs(disc.InternalDelay[0]-0.5) > 1e-9 {
		t.Errorf("InternalDelay[0] = %v, want 0.5", disc.InternalDelay[0])
	}
	if len(disc.OutputDelay) != 1 || disc.OutputDelay[0] != 3 {
		t.Errorf("OutputDelay = %v, want [3]", disc.OutputDelay)
	}
}

func TestC2DDelayModelingInternalMIMO(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	sys.InputDelay = []float64{0.35, 0.2}

	dt := 0.1
	disc, err := sys.DiscretizeWithOpts(dt, C2DOptions{Method: "zoh", DelayModeling: "internal"})
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := disc.Dims()
	if n != 2 {
		t.Errorf("state dim = %d, want 2", n)
	}
	if m != 2 || p != 2 {
		t.Errorf("dims m=%d p=%d, want 2,2", m, p)
	}
	if len(disc.InternalDelay) != 1 {
		t.Fatalf("InternalDelay len = %d, want 1 (only ch0 has fractional)", len(disc.InternalDelay))
	}
	if math.Abs(disc.InternalDelay[0]-0.5) > 1e-9 {
		t.Errorf("InternalDelay[0] = %v, want 0.5", disc.InternalDelay[0])
	}
	if disc.InputDelay[0] != 3 {
		t.Errorf("InputDelay[0] = %v, want 3", disc.InputDelay[0])
	}
	if disc.InputDelay[1] != 2 {
		t.Errorf("InputDelay[1] = %v, want 2", disc.InputDelay[1])
	}
}
