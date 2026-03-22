package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestH2Norm_SISO_1x1(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	want := math.Sqrt(0.5)
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("H2 = %g, want %g", got, want)
	}
}

func TestH2Norm_2x2_NonSymA(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-2, 1, 0, -3}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if got <= 0 || math.IsNaN(got) {
		t.Errorf("H2 = %g, want positive", got)
	}
}

func TestH2Norm_MIMO(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, nil), 0)

	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if got <= 0 {
		t.Errorf("H2 = %g, want positive", got)
	}
}

func TestH2Norm_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0.1)

	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if got <= 0 {
		t.Errorf("H2 = %g, want positive", got)
	}
}

func TestH2Norm_Discrete_WithD(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{2}), 0.1)

	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	gotNoD, _ := H2Norm(&System{
		A: mat.NewDense(1, 1, []float64{0.5}),
		B: mat.NewDense(1, 1, []float64{1}),
		C: mat.NewDense(1, 1, []float64{1}),
		D: mat.NewDense(1, 1, []float64{0}),
		Dt: 0.1,
	})
	if got <= gotNoD {
		t.Errorf("H2 with D (%g) should be > H2 without D (%g)", got, gotNoD)
	}
}

func TestH2Norm_Continuous_DNonZero(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), 0)

	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(got, 1) {
		t.Errorf("H2 = %g, want +Inf", got)
	}
}

func TestH2Norm_Unstable(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	_, err := H2Norm(sys)
	if !errors.Is(err, ErrUnstable) {
		t.Errorf("got %v, want ErrUnstable", err)
	}
}

func TestH2Norm_PureGain_Continuous(t *testing.T) {
	sys, _ := New(nil, nil, nil, mat.NewDense(1, 1, []float64{5}), 0)
	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(got, 1) {
		t.Errorf("H2 = %g, want +Inf", got)
	}
}

func TestH2Norm_PureGain_Discrete(t *testing.T) {
	sys, _ := New(nil, nil, nil, mat.NewDense(2, 2, []float64{3, 0, 0, 4}), 0.1)
	got, err := H2Norm(sys)
	if err != nil {
		t.Fatal(err)
	}
	want := 5.0
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("H2 = %g, want %g", got, want)
	}
}

func TestHSV_SISO_1x1(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	hsv, err := HSV(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(hsv) != 1 {
		t.Fatalf("len(hsv) = %d, want 1", len(hsv))
	}
	want := 0.5
	if math.Abs(hsv[0]-want) > 1e-10 {
		t.Errorf("hsv[0] = %g, want %g", hsv[0], want)
	}
}

func TestHSV_Diagonal(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -3}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, nil), 0)

	hsv, err := HSV(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(hsv) != 2 {
		t.Fatalf("len = %d", len(hsv))
	}
	if hsv[0] < hsv[1] {
		t.Errorf("not descending: %v", hsv)
	}

	want0 := 0.5
	want1 := 1.0 / 6.0
	if math.Abs(hsv[0]-want0) > 1e-10 {
		t.Errorf("hsv[0] = %g, want %g", hsv[0], want0)
	}
	if math.Abs(hsv[1]-want1) > 1e-10 {
		t.Errorf("hsv[1] = %g, want %g", hsv[1], want1)
	}
}

func TestHSV_MIMO(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, nil), 0)

	hsv, err := HSV(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(hsv) != 2 {
		t.Fatalf("len = %d", len(hsv))
	}
	if hsv[0] < hsv[1] {
		t.Errorf("not descending: %v", hsv)
	}
}

func TestHSV_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0.5, 0.1, 0, 0.8}),
		mat.NewDense(2, 1, []float64{1, 0.5}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0.1)

	hsv, err := HSV(sys)
	if err != nil {
		t.Fatal(err)
	}
	if len(hsv) != 2 {
		t.Fatalf("len = %d", len(hsv))
	}
	for _, v := range hsv {
		if v < 0 || math.IsNaN(v) {
			t.Errorf("invalid hsv: %v", hsv)
		}
	}
}

func TestHSV_Unstable(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	_, err := HSV(sys)
	if !errors.Is(err, ErrUnstable) {
		t.Errorf("got %v, want ErrUnstable", err)
	}
}

func TestHSV_Empty(t *testing.T) {
	sys, _ := New(nil, nil, nil, mat.NewDense(1, 1, []float64{1}), 0)
	hsv, err := HSV(sys)
	if err != nil {
		t.Fatal(err)
	}
	if hsv != nil {
		t.Errorf("hsv = %v, want nil", hsv)
	}
}

func TestHinfNorm_FirstOrder(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	norm, omega, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(norm-1.0) > 1e-6 {
		t.Errorf("Hinf = %g, want 1.0", norm)
	}
	if omega > 0.1 {
		t.Errorf("omega = %g, want ~0 (DC peak)", omega)
	}
}

func TestHinfNorm_SecondOrder_Underdamped(t *testing.T) {
	wn := 10.0
	zeta := 0.1
	A := mat.NewDense(2, 2, []float64{
		0, 1,
		-wn * wn, -2 * zeta * wn,
	})
	B := mat.NewDense(2, 1, []float64{0, wn * wn})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)

	norm, omega, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}

	wPeak := wn * math.Sqrt(1-2*zeta*zeta)
	expected := 1.0 / (2 * zeta * math.Sqrt(1-zeta*zeta))
	if math.Abs(norm-expected)/expected > 1e-4 {
		t.Errorf("Hinf = %g, want ~%g", norm, expected)
	}
	if math.Abs(omega-wPeak)/wPeak > 0.1 {
		t.Errorf("omega = %g, want ~%g", omega, wPeak)
	}
}

func TestHinfNorm_WithD(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.5}), 0)

	norm, _, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if norm < 1.5-0.01 {
		t.Errorf("Hinf = %g, want >= 1.5 (DC gain = 1+0.5)", norm)
	}
}

func TestHinfNorm_MIMO(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, nil), 0)

	norm, _, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if norm <= 0 {
		t.Errorf("Hinf = %g, want positive", norm)
	}
}

func TestHinfNorm_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0.1)

	norm, _, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if norm <= 0 {
		t.Errorf("Hinf = %g, want positive", norm)
	}
}

func TestHinfNorm_Unstable(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	_, _, err := HinfNorm(sys)
	if !errors.Is(err, ErrUnstable) {
		t.Errorf("got %v, want ErrUnstable", err)
	}
}

func TestHinfNorm_PureGain(t *testing.T) {
	sys, _ := New(nil, nil, nil, mat.NewDense(2, 2, []float64{3, 0, 0, 4}), 0)
	norm, omega, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(norm-4.0) > 1e-10 {
		t.Errorf("Hinf = %g, want 4.0", norm)
	}
	if omega != 0 {
		t.Errorf("omega = %g, want 0", omega)
	}
}

func TestHinfNorm_MatchesBode(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 2, 0, -3}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)

	norm, _, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}

	maxBode := 0.0
	for i := range 1000 {
		w := 0.001 * math.Pow(1000000, float64(i)/999)
		G, _ := sys.EvalFr(complex(0, w))
		mag := cmplx.Abs(G[0][0])
		if mag > maxBode {
			maxBode = mag
		}
	}

	if norm < maxBode*0.999 {
		t.Errorf("Hinf(%g) < max Bode(%g)", norm, maxBode)
	}
	if norm > maxBode*1.01 {
		t.Errorf("Hinf(%g) >> max Bode(%g), tolerance exceeded", norm, maxBode)
	}
}

func TestH2_HSV_Relationship(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)

	h2, _ := H2Norm(sys)
	hsv, _ := HSV(sys)
	hinf, _, _ := HinfNorm(sys)

	if h2 <= 0 || hinf <= 0 {
		t.Skipf("h2=%g hinf=%g", h2, hinf)
	}
	if len(hsv) == 0 {
		t.Skip("no HSV")
	}
	if hsv[0] <= 0 {
		t.Skip("zero HSV")
	}

	if hinf < hsv[0]*0.99 {
		t.Errorf("Hinf(%g) < hsv[0](%g), Hinf >= largest HSV expected", hinf, hsv[0])
	}
}
