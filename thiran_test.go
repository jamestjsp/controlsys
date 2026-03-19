package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

func TestThiranDelayOrder1(t *testing.T) {
	dt := 0.1
	tau := 0.15 // 1.5 samples
	sys, err := ThiranDelay(tau, 1, dt)
	if err != nil {
		t.Fatal(err)
	}
	if !sys.IsDiscrete() {
		t.Error("should be discrete")
	}
	n, m, p := sys.Dims()
	if m != 1 || p != 1 {
		t.Errorf("dims = (%d,%d,%d), want (_,1,1)", n, m, p)
	}
}

func TestThiranDelayAllpass(t *testing.T) {
	dt := 1.0
	D := 3.4 // 3.4 samples
	sys, err := ThiranDelay(D*dt, 3, dt)
	if err != nil {
		t.Fatal(err)
	}

	tfRes, _ := sys.TransferFunction(nil)

	for _, w := range []float64{0.01, 0.1, 0.5, 1.0, 2.0} {
		z := cmplx.Exp(complex(0, w*dt))
		h := tfRes.TF.Eval(z)[0][0]
		mag := cmplx.Abs(h)
		if math.Abs(mag-1) > 1e-8 {
			t.Errorf("w=%v: |H| = %v, want 1 (allpass)", w, mag)
		}
	}
}

func TestThiranDelayGroupDelay(t *testing.T) {
	dt := 1.0
	D := 2.7
	order := 3
	sys, err := ThiranDelay(D*dt, order, dt)
	if err != nil {
		t.Fatal(err)
	}

	tfRes, _ := sys.TransferFunction(nil)

	// Group delay at DC should approximate D samples
	dw := 1e-6
	z1 := cmplx.Exp(complex(0, dw))
	z2 := cmplx.Exp(complex(0, 2*dw))
	h1 := tfRes.TF.Eval(z1)[0][0]
	h2 := tfRes.TF.Eval(z2)[0][0]
	phase1 := cmplx.Phase(h1)
	phase2 := cmplx.Phase(h2)
	groupDelay := -(phase2 - phase1) / dw

	if math.Abs(groupDelay-D) > 0.01 {
		t.Errorf("group delay at DC = %v, want %v", groupDelay, D)
	}
}

func TestThiranDelayIntegerFallback(t *testing.T) {
	dt := 0.1
	tau := 0.5 // exactly 5 samples
	sys, err := ThiranDelay(tau, 1, dt)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := sys.Dims()
	if n != 5 {
		t.Errorf("integer delay: state dim = %d, want 5", n)
	}

	// Should act as pure z^{-5}
	tfRes, _ := sys.TransferFunction(nil)
	z := cmplx.Exp(complex(0, 0.3))
	h := tfRes.TF.Eval(z)[0][0]
	expected := cmplx.Pow(z, -5)
	if cmplx.Abs(h-expected) > 1e-10 {
		t.Errorf("integer delay: H(z)=%v, want z^{-5}=%v", h, expected)
	}
}

func TestThiranDelayNegative(t *testing.T) {
	_, err := ThiranDelay(-1, 1, 0.1)
	if !errors.Is(err, ErrNegativeDelay) {
		t.Errorf("expected ErrNegativeDelay, got %v", err)
	}
}

func TestThiranDelayInvalidOrder(t *testing.T) {
	_, err := ThiranDelay(0.5, 0, 0.1)
	if err == nil {
		t.Error("order 0 should error")
	}
	_, err = ThiranDelay(0.5, 11, 0.1)
	if err == nil {
		t.Error("order 11 should error")
	}
}

func TestThiranDelayStability(t *testing.T) {
	dt := 1.0
	for _, D := range []float64{0.6, 1.5, 2.7, 3.7, 5.1} {
		order := int(math.Floor(D))
		if order < 1 {
			order = 1
		}
		if order > 10 {
			order = 10
		}
		sys, err := ThiranDelay(D*dt, order, dt)
		if err != nil {
			t.Fatalf("D=%v order=%d: %v", D, order, err)
		}
		stable, err := sys.IsStable()
		if err != nil {
			t.Fatalf("D=%v order=%d: stability check failed: %v", D, order, err)
		}
		if !stable {
			t.Errorf("D=%v order=%d: Thiran should be stable", D, order)
		}
	}
}

// Julia: thiran(2*Ts, Ts) == 1/z^2 for integer delays
func TestThiranIntegerDelayExact(t *testing.T) {
	for _, nSamples := range []int{1, 2, 3, 5} {
		for _, dt := range []float64{1.0, 0.5, 1.1} {
			tau := float64(nSamples) * dt
			sys, err := ThiranDelay(tau, 1, dt)
			if err != nil {
				t.Fatalf("D=%d dt=%v: %v", nSamples, dt, err)
			}
			tfRes, _ := sys.TransferFunction(nil)

			for _, w := range []float64{0.1, 0.5, 1.0, 2.0} {
				z := cmplx.Exp(complex(0, w*dt))
				got := tfRes.TF.Eval(z)[0][0]
				want := cmplx.Pow(z, complex(float64(-nSamples), 0))
				if cmplx.Abs(got-want) > 1e-10 {
					t.Errorf("D=%d dt=%v w=%v: got %v, want z^{-%d}=%v", nSamples, dt, w, got, nSamples, want)
				}
			}
		}
	}
}

// Julia: thiran(pi, 1) uses order=ceil(pi)=4 internally with different stability bounds.
// We use order=3 since our stability condition is D >= N-0.5 (π≈3.14 >= 2.5).
// Verify allpass + group delay properties match the fractional delay.
func TestThiranPiDelay(t *testing.T) {
	D := math.Pi
	dt := 1.0
	sys, err := ThiranDelay(D*dt, 3, dt)
	if err != nil {
		t.Fatal(err)
	}

	tfRes, _ := sys.TransferFunction(nil)

	for _, w := range []float64{0.01, 0.1, 0.5, 1.0} {
		z := cmplx.Exp(complex(0, w*dt))
		h := tfRes.TF.Eval(z)[0][0]

		// Allpass: |H(z)| = 1
		mag := cmplx.Abs(h)
		if math.Abs(mag-1) > 1e-8 {
			t.Errorf("w=%v: |H| = %v, want 1", w, mag)
		}
	}

	// Group delay at DC should ≈ π samples
	dw := 1e-6
	z1 := cmplx.Exp(complex(0, dw))
	z2 := cmplx.Exp(complex(0, 2*dw))
	h1 := tfRes.TF.Eval(z1)[0][0]
	h2 := tfRes.TF.Eval(z2)[0][0]
	gd := -(cmplx.Phase(h2) - cmplx.Phase(h1)) / dw
	if math.Abs(gd-D) > 0.01 {
		t.Errorf("group delay = %v, want π ≈ %v", gd, D)
	}
}

// Verify Thiran coefficients directly against Julia's known values.
// Julia: thiran(pi, 1) with order=4 has specific coefficients.
// We compute order=4 coefficients directly to cross-validate our coefficient formula.
func TestThiranCoeffsDirectly(t *testing.T) {
	D := math.Pi
	N := 4
	a := thiranCoeffs(D, N)

	wantA := []float64{
		1.0,
		0.8290601401044773,
		-0.03424682772398137,
		0.0042438423976339556,
		-0.00031815668236122736,
	}

	for k := 0; k <= N; k++ {
		if math.Abs(a[k]-wantA[k]) > 1e-12 {
			t.Errorf("a[%d] = %.16g, want %.16g", k, a[k], wantA[k])
		}
	}
}

// Verify integerDelaySS frequency response matches exp(-jω*d*dt)
func TestIntegerDelaySSFreqResponse(t *testing.T) {
	for _, d := range []int{1, 3, 7} {
		dt := 0.1
		sys, _ := integerDelaySS(d, dt)
		tfRes, _ := sys.TransferFunction(nil)

		for _, w := range []float64{0.5, 1.0, 5.0} {
			z := cmplx.Exp(complex(0, w*dt))
			got := tfRes.TF.Eval(z)[0][0]
			want := cmplx.Exp(complex(0, -w*float64(d)*dt))
			if cmplx.Abs(got-want) > 1e-12 {
				t.Errorf("d=%d w=%v: got %v, want %v", d, w, got, want)
			}
		}
	}
}

func TestThiranDelayTooSmall(t *testing.T) {
	_, err := ThiranDelay(0.01, 3, 1.0)
	if err == nil {
		t.Error("delay 0.01 samples with order 3 should fail (D < N-0.5)")
	}
}

func TestThiranDelayInvalidDt(t *testing.T) {
	_, err := ThiranDelay(1.0, 1, 0)
	if !errors.Is(err, ErrInvalidSampleTime) {
		t.Errorf("expected ErrInvalidSampleTime, got %v", err)
	}
	_, err = ThiranDelay(1.0, 1, -1)
	if !errors.Is(err, ErrInvalidSampleTime) {
		t.Errorf("expected ErrInvalidSampleTime, got %v", err)
	}
}
