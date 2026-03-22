package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestDCGain_SISO_Continuous(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(g.At(0, 0)-1.0) > 1e-12 {
		t.Errorf("got %f, want 1.0", g.At(0, 0))
	}
}

func TestDCGain_SISO_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		1.0,
	)
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(g.At(0, 0)-2.0) > 1e-12 {
		t.Errorf("got %f, want 2.0", g.At(0, 0))
	}
}

func TestDCGain_MIMO_NonSymmetricA(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-2, 1, 0, -3}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	// G(0) = -C*inv(A)*B = -inv(A) since C=I, B=I
	// A = [-2 1; 0 -3], inv(A) = [-1/2 -1/6; 0 -1/3]
	// G(0) = [1/2 1/6; 0 1/3]
	want := mat.NewDense(2, 2, []float64{0.5, 1.0 / 6, 0, 1.0 / 3})
	if !matEqual(g, want, 1e-12) {
		t.Errorf("got\n%v\nwant\n%v", mat.Formatted(g), mat.Formatted(want))
	}
}

func TestDCGain_PureGain(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6}), 0)
	g, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	want := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	if !matEqual(g, want, 1e-15) {
		t.Errorf("DCGain of pure gain != D")
	}
}

func TestDCGain_Integrator_Error(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_, err := sys.DCGain()
	if err == nil {
		t.Fatal("expected error for integrator (singular A)")
	}
}

func TestDCGain_Discrete_PoleAtOne_Error(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_, err := sys.DCGain()
	if err == nil {
		t.Fatal("expected error for discrete pole at z=1")
	}
}

func TestDamp_ContinuousUnderdamped(t *testing.T) {
	wn := 10.0
	zeta := 0.3
	// Poles: -zeta*wn ± j*wn*sqrt(1-zeta^2)
	sigma := -zeta * wn
	wd := wn * math.Sqrt(1-zeta*zeta)
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -(wn * wn), 2 * sigma}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	info, err := Damp(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(info) != 2 {
		t.Fatalf("got %d poles, want 2", len(info))
	}

	for _, d := range info {
		if math.Abs(d.Wn-wn) > 1e-10 {
			t.Errorf("Wn = %f, want %f", d.Wn, wn)
		}
		if math.Abs(d.Zeta-zeta) > 1e-10 {
			t.Errorf("Zeta = %f, want %f", d.Zeta, zeta)
		}
		wantTau := 1.0 / (zeta * wn)
		if math.Abs(d.Tau-wantTau) > 1e-10 {
			t.Errorf("Tau = %f, want %f", d.Tau, wantTau)
		}
	}

	_ = wd
}

func TestDamp_Discrete(t *testing.T) {
	dt := 0.01
	wnCont := 10.0
	zetaCont := 0.5

	sigma := -zetaCont * wnCont
	wd := wnCont * math.Sqrt(1-zetaCont*zetaCont)
	p := cmplx.Exp(complex(sigma, wd) * complex(dt, 0))

	sys, _ := New(
		mat.NewDense(2, 2, []float64{
			real(p + cmplx.Conj(p)), -cmplx.Abs(p) * cmplx.Abs(p),
			1, 0,
		}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 1, []float64{0}),
		dt,
	)

	info, err := Damp(sys)
	if err != nil {
		t.Fatal(err)
	}

	for _, d := range info {
		if math.Abs(d.Wn-wnCont) > 0.5 {
			t.Errorf("Wn = %f, want ~%f", d.Wn, wnCont)
		}
		if math.Abs(d.Zeta-zetaCont) > 0.05 {
			t.Errorf("Zeta = %f, want ~%f", d.Zeta, zetaCont)
		}
	}
}

func TestDamp_PureGain(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(1, 1, []float64{5}), 0)
	info, err := Damp(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(info) != 0 {
		t.Errorf("expected empty DampInfo for pure gain, got %d", len(info))
	}
}

func TestStep_SISO_Continuous(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	resp, err := Step(sys, 5.0)
	if err != nil {
		t.Fatal(err)
	}

	_, steps := resp.Y.Dims()
	for k := 0; k < steps; k++ {
		tk := resp.T[k]
		want := 1 - math.Exp(-tk)
		got := resp.Y.At(0, k)
		if math.Abs(got-want) > 0.01 {
			t.Errorf("t=%.3f: got %f, want %f", tk, got, want)
		}
	}
}

func TestStep_Discrete_Integrator(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		1.0,
	)

	resp, err := Step(sys, 4.0)
	if err != nil {
		t.Fatal(err)
	}

	for k := 0; k < 5; k++ {
		got := resp.Y.At(0, k)
		want := float64(k)
		if math.Abs(got-want) > 1e-12 {
			t.Errorf("k=%d: got %f, want %f", k, got, want)
		}
	}
}

func TestStep_MIMO_NonSymmetricA(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-2, 1, 0, -3}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)

	resp, err := Step(sys, 3.0)
	if err != nil {
		t.Fatal(err)
	}

	rows, steps := resp.Y.Dims()
	if rows != 4 {
		t.Fatalf("Y rows = %d, want 4 (p*m=2*2)", rows)
	}

	dcGain, _ := sys.DCGain()
	lastK := steps - 1
	for j := 0; j < 2; j++ {
		for i := 0; i < 2; i++ {
			got := resp.Y.At(j*2+i, lastK)
			want := dcGain.At(i, j)
			if math.Abs(got-want) > 0.05 {
				t.Errorf("steady-state Y[%d,%d]=%f, want DC gain %f", i, j, got, want)
			}
		}
	}
}

func TestStep_PureGain(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(1, 1, []float64{3}), 0)
	resp, err := Step(sys, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	_, steps := resp.Y.Dims()
	for k := 0; k < steps; k++ {
		if math.Abs(resp.Y.At(0, k)-3.0) > 1e-12 {
			t.Errorf("k=%d: got %f, want 3.0", k, resp.Y.At(0, k))
		}
	}
}

func TestImpulse_SISO_Continuous(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	resp, err := Impulse(sys, 5.0)
	if err != nil {
		t.Fatal(err)
	}

	_, steps := resp.Y.Dims()
	for k := 1; k < steps; k++ {
		tk := resp.T[k]
		want := math.Exp(-tk)
		got := resp.Y.At(0, k)
		if math.Abs(got-want) > 0.05 {
			t.Errorf("t=%.3f: got %f, want %f", tk, got, want)
		}
	}
}

func TestImpulse_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		1.0,
	)

	resp, err := Impulse(sys, 4.0)
	if err != nil {
		t.Fatal(err)
	}

	// y[0] = C*B*1 = 1*1*1 = 1 (B*impulse at k=0, then propagate through C)
	// Actually: y[0] = C*x0 + D*u[0] = 0 + 0 = 0... wait
	// x0 = nil → zero. y[0] = D*u[0] = 0. x[1] = A*0 + B*1 = 1
	// y[1] = C*x[1] + D*u[1] = 1 + 0 = 1
	// y[2] = C*(A*1 + B*0) = 0.5
	if math.Abs(resp.Y.At(0, 0)-0) > 1e-12 {
		t.Errorf("y[0] = %f, want 0 (D=0, x0=0)", resp.Y.At(0, 0))
	}
	if math.Abs(resp.Y.At(0, 1)-1) > 1e-12 {
		t.Errorf("y[1] = %f, want 1", resp.Y.At(0, 1))
	}
	if math.Abs(resp.Y.At(0, 2)-0.5) > 1e-12 {
		t.Errorf("y[2] = %f, want 0.5", resp.Y.At(0, 2))
	}
}

func TestInitial_SISO_Continuous(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	x0 := mat.NewVecDense(1, []float64{1})
	resp, err := Initial(sys, x0, 5.0)
	if err != nil {
		t.Fatal(err)
	}

	_, steps := resp.Y.Dims()
	for k := 0; k < steps; k++ {
		tk := resp.T[k]
		want := math.Exp(-tk)
		got := resp.Y.At(0, k)
		if math.Abs(got-want) > 0.01 {
			t.Errorf("t=%.3f: got %f, want %f", tk, got, want)
		}
	}
}

func TestInitial_NilX0_Error(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_, err := Initial(sys, nil, 1.0)
	if err == nil {
		t.Fatal("expected error for nil x0")
	}
}

func TestStep_AutoTFinal(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	resp, err := Step(sys, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.T) < 50 {
		t.Errorf("auto tFinal produced only %d points, expected more", len(resp.T))
	}
	last := resp.Y.At(0, len(resp.T)-1)
	if math.Abs(last-1.0) > 0.01 {
		t.Errorf("final value %f not near steady-state 1.0", last)
	}
}
