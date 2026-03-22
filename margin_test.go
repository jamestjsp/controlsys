package controlsys

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// === Phase 1: helper tests ===

func TestFindCrossings_Basic(t *testing.T) {
	omega := []float64{0.1, 1, 10, 100}
	vals := []float64{-2, -1, 0.5, 2}
	cs := findCrossings(omega, vals, 0)
	if len(cs) != 1 {
		t.Fatalf("got %d crossings, want 1", len(cs))
	}
	if cs[0].idx != 1 {
		t.Errorf("idx = %d, want 1", cs[0].idx)
	}
	if cs[0].w < 1 || cs[0].w > 10 {
		t.Errorf("w = %v, want between 1 and 10", cs[0].w)
	}
}

func TestFindCrossings_Multiple(t *testing.T) {
	omega := []float64{0.1, 1, 10, 100, 1000}
	vals := []float64{-1, 1, -1, 1, -1}
	cs := findCrossings(omega, vals, 0)
	if len(cs) != 4 {
		t.Fatalf("got %d crossings, want 4", len(cs))
	}
}

func TestFindCrossings_NoCrossing(t *testing.T) {
	omega := []float64{0.1, 1, 10}
	vals := []float64{1, 2, 3}
	cs := findCrossings(omega, vals, 0)
	if len(cs) != 0 {
		t.Fatalf("got %d crossings, want 0", len(cs))
	}
}

func TestFindCrossings_InfNaN(t *testing.T) {
	omega := []float64{0.1, 1, 10}
	vals := []float64{math.Inf(-1), 0, 1}
	cs := findCrossings(omega, vals, 0)
	if len(cs) != 0 {
		t.Fatalf("got %d crossings, want 0 (Inf skipped)", len(cs))
	}
}

func TestPhaseCrossings_SimpleWrap(t *testing.T) {
	omega := []float64{1, 2, 3, 4}
	phase := []float64{-170, -175, -185, -190}
	cs := phaseCrossings(omega, phase, -180)
	if len(cs) != 1 {
		t.Fatalf("got %d crossings, want 1", len(cs))
	}
	if cs[0].idx != 1 {
		t.Errorf("idx = %d, want 1", cs[0].idx)
	}
}

func TestPhaseCrossings_MultipleRevolutions(t *testing.T) {
	omega := []float64{1, 2, 3, 4, 5, 6}
	phase := []float64{-170, -190, -350, -370, -530, -550}
	cs := phaseCrossings(omega, phase, -180)
	if len(cs) != 3 {
		t.Fatalf("got %d crossings, want 3", len(cs))
	}
}

func TestRefineCrossing_Precision(t *testing.T) {
	// f(w) = 1/sqrt(1+w^2) - 1/sqrt(2), zero at w=1
	f := func(w float64) float64 {
		return 1/math.Sqrt(1+w*w) - 1/math.Sqrt(2)
	}
	w := refineCrossing(0.1, 10, f)
	if math.Abs(w-1) > 1e-8 {
		t.Errorf("refined crossing = %v, want 1.0", w)
	}
}

// === Phase 2: Margin / AllMargin tests ===

// G(s) = 1/s: integrator
// PM = 90 deg at w = 1, GM = +Inf
func TestMargin_Integrator(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if !math.IsInf(r.GainMargin, 1) {
		t.Errorf("GM = %v, want +Inf", r.GainMargin)
	}
	if math.Abs(r.PhaseMargin-90) > 1 {
		t.Errorf("PM = %v, want ~90 deg", r.PhaseMargin)
	}
	if !math.IsNaN(r.WpFreq) {
		t.Errorf("WpFreq = %v, want NaN", r.WpFreq)
	}
	if math.Abs(r.WgFreq-1) > 0.05 {
		t.Errorf("WgFreq = %v, want ~1", r.WgFreq)
	}
}

// G(s) = 1/(s(s+1)(s+2)):
// GM = 20*log10(6) ≈ 15.56 dB at w_pc = sqrt(2)
// PM ≈ 53.4 deg at w_gc ≈ 0.446
func TestMargin_ThirdOrder(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			0, 1, 0,
			0, 0, 1,
			0, -2, -3,
		}),
		mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 0, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	wantGM := 20 * math.Log10(6)
	if math.Abs(r.GainMargin-wantGM) > 0.5 {
		t.Errorf("GM = %v dB, want ~%v dB", r.GainMargin, wantGM)
	}
	if math.Abs(r.WpFreq-math.Sqrt(2)) > 0.05 {
		t.Errorf("WpFreq = %v, want ~%v", r.WpFreq, math.Sqrt(2))
	}
	if math.Abs(r.PhaseMargin-53.4) > 1.5 {
		t.Errorf("PM = %v deg, want ~53.4 deg", r.PhaseMargin)
	}
	if math.Abs(r.WgFreq-0.446) > 0.05 {
		t.Errorf("WgFreq = %v, want ~0.446", r.WgFreq)
	}
}

// G(s) = 100/(s^3 + 11s^2 + 10s) = 10/(s(s+1)(0.1s+1))
// GM ≈ 0.83 dB at w_pc = sqrt(10)
// This system is very close to instability
func TestMargin_NearUnstable(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			0, 1, 0,
			0, 0, 1,
			0, -10, -11,
		}),
		mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{100, 0, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	// GM = -20*log10(10/11) = 20*log10(11/10) ≈ 0.828 dB
	wantGM := 20 * math.Log10(11.0/10.0)
	if math.Abs(r.GainMargin-wantGM) > 0.3 {
		t.Errorf("GM = %v dB, want ~%v dB", r.GainMargin, wantGM)
	}
	if math.Abs(r.WpFreq-math.Sqrt(10)) > 0.1 {
		t.Errorf("WpFreq = %v, want ~%v", r.WpFreq, math.Sqrt(10))
	}
}

// G = 0.1/(s+1): |G| < 0dB everywhere -> no crossings -> infinite margins
func TestMargin_LowGain(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if !math.IsInf(r.GainMargin, 1) {
		t.Errorf("GM = %v, want +Inf", r.GainMargin)
	}
	if !math.IsInf(r.PhaseMargin, 1) {
		t.Errorf("PM = %v, want +Inf", r.PhaseMargin)
	}
}

func TestMargin_GainOnly(t *testing.T) {
	sys, err := NewGain(mat.NewDense(1, 1, []float64{5}), 0)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if !math.IsInf(r.GainMargin, 1) {
		t.Errorf("GM = %v, want +Inf", r.GainMargin)
	}
	if !math.IsInf(r.PhaseMargin, 1) {
		t.Errorf("PM = %v, want +Inf", r.PhaseMargin)
	}
}

func TestMargin_MIMOReject(t *testing.T) {
	sys, err := NewGain(mat.NewDense(2, 2, []float64{1, 0, 0, 1}), 0)
	if err != nil {
		t.Fatal(err)
	}

	_, err = Margin(sys)
	if err == nil {
		t.Fatal("expected error for MIMO system")
	}
}

func TestAllMargin_ThirdOrder(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			0, 1, 0,
			0, 0, 1,
			0, -2, -3,
		}),
		mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 0, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	all, err := AllMargin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if len(all.GainCrossFreqs) != 1 {
		t.Fatalf("got %d gain crossovers, want 1", len(all.GainCrossFreqs))
	}
	if len(all.PhaseCrossFreqs) != 1 {
		t.Fatalf("got %d phase crossovers, want 1", len(all.PhaseCrossFreqs))
	}
}

// Discrete: discretize the integrator, verify PM close to continuous
func TestMargin_Discrete(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	dt := 0.01
	dsys, err := sys.DiscretizeZOH(dt)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(dsys)
	if err != nil {
		t.Fatal(err)
	}

	if math.Abs(r.PhaseMargin-90) > 3 {
		t.Errorf("discrete PM = %v, want ~90 deg", r.PhaseMargin)
	}
}

// System with IO delay: G(s) = 2*exp(-s)/(s+1)
// w_gc = sqrt(3), PM = 180 - arctan(sqrt(3))*180/pi - sqrt(3)*180/pi ≈ 20.76 deg
func TestMargin_WithDelay(t *testing.T) {
	sys, err := NewWithDelay(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	wantPM := 180 - math.Atan(math.Sqrt(3))*180/math.Pi - math.Sqrt(3)*180/math.Pi
	if math.Abs(r.PhaseMargin-wantPM) > 2 {
		t.Errorf("PM = %v, want ~%v", r.PhaseMargin, wantPM)
	}
	if math.Abs(r.WgFreq-math.Sqrt(3)) > 0.1 {
		t.Errorf("WgFreq = %v, want ~%v", r.WgFreq, math.Sqrt(3))
	}
}

// === Phase 3: Bandwidth tests ===

// G(s) = 1/(s+1): BW = 1 rad/s at -3dB
func TestBandwidth_FirstOrder(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	bw, err := Bandwidth(sys, 0)
	if err != nil {
		t.Fatal(err)
	}

	if math.Abs(bw-1.0) > 0.05 {
		t.Errorf("bandwidth = %v, want ~1.0", bw)
	}
}

// G(s) = wn^2/(s^2+2*z*wn*s+wn^2), wn=10, z=0.7
// |G(jw)| = wn^2 / sqrt((wn^2-w^2)^2 + (2*z*wn*w)^2)
func TestBandwidth_SecondOrder(t *testing.T) {
	wn := 10.0
	z := 0.7
	sys, err := New(
		mat.NewDense(2, 2, []float64{
			0, 1,
			-wn * wn, -2 * z * wn,
		}),
		mat.NewDense(2, 1, []float64{0, wn * wn}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	bw, err := Bandwidth(sys, 0)
	if err != nil {
		t.Fatal(err)
	}

	// BW for 2nd order: w_bw = wn * sqrt(1 - 2z^2 + sqrt(4z^4 - 4z^2 + 2))
	inner := 4*z*z*z*z - 4*z*z + 2
	wBW := wn * math.Sqrt(1-2*z*z+math.Sqrt(inner))
	if math.Abs(bw-wBW) > 0.5 {
		t.Errorf("bandwidth = %v, want ~%v", bw, wBW)
	}
}

func TestBandwidth_CustomDrop(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	bw, err := Bandwidth(sys, -6)
	if err != nil {
		t.Fatal(err)
	}

	// -6dB: 1/sqrt(1+w^2) = 10^(-6/20) = 0.5012 -> w = sqrt(1/0.5012^2 - 1) ≈ 1.732
	want := math.Sqrt(1/math.Pow(10, -6.0/20)*1/math.Pow(10, -6.0/20) - 1)
	// Simpler: |G|=10^(-6/20), 1/(1+w^2) = 10^(-6/10), w^2 = 10^(6/10)-1
	w2 := math.Pow(10, 0.6) - 1
	want = math.Sqrt(w2)
	if math.Abs(bw-want) > 0.1 {
		t.Errorf("bandwidth(-6dB) = %v, want ~%v", bw, want)
	}
}

func TestBandwidth_GainOnly(t *testing.T) {
	sys, err := NewGain(mat.NewDense(1, 1, []float64{5}), 0)
	if err != nil {
		t.Fatal(err)
	}

	bw, err := Bandwidth(sys, 0)
	if err != nil {
		t.Fatal(err)
	}

	if !math.IsInf(bw, 1) {
		t.Errorf("bandwidth = %v, want +Inf (constant gain)", bw)
	}
}

func TestBandwidth_Discrete(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	dt := 0.01
	dsys, err := sys.DiscretizeZOH(dt)
	if err != nil {
		t.Fatal(err)
	}

	bw, err := Bandwidth(dsys, 0)
	if err != nil {
		t.Fatal(err)
	}

	if math.Abs(bw-1.0) > 0.1 {
		t.Errorf("discrete bandwidth = %v, want ~1.0", bw)
	}
}

// === Phase 4: DiskMargin tests ===

// L(s) = 10/(s+1): S = (s+1)/(s+11), Ms ≈ 1.0
func TestDiskMargin_StableFirstOrder(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{10}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	dm, err := DiskMargin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if dm.PeakSensitivity > 1.1 {
		t.Errorf("Ms = %v, want ≤ ~1.0", dm.PeakSensitivity)
	}
	if dm.Alpha < 0.9 {
		t.Errorf("Alpha = %v, want ~1.0", dm.Alpha)
	}
	if dm.PhaseMargin < 50 {
		t.Errorf("disk PM = %v, want ≥ 50 deg", dm.PhaseMargin)
	}
}

// L(s) = 10/(s(s+1)): moderately damped, Ms > 1
func TestDiskMargin_ModerateLoop(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{
			0, 1,
			0, -1,
		}),
		mat.NewDense(2, 1, []float64{0, 10}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	dm, err := DiskMargin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if dm.PeakSensitivity < 1 {
		t.Errorf("Ms = %v, want > 1", dm.PeakSensitivity)
	}
	if dm.Alpha <= 0 || dm.Alpha >= 1 {
		t.Errorf("Alpha = %v, want in (0, 1)", dm.Alpha)
	}
	if dm.PhaseMargin <= 0 {
		t.Errorf("disk PM = %v, want > 0", dm.PhaseMargin)
	}
	if dm.GainMargin[0] <= 0 || dm.GainMargin[0] >= 1 {
		t.Errorf("GM_low = %v, want in (0,1)", dm.GainMargin[0])
	}
	if dm.GainMargin[1] <= 1 {
		t.Errorf("GM_high = %v, want > 1", dm.GainMargin[1])
	}
}

// Verify disk margin is tighter than classical margins
func TestDiskMargin_VsClassical(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			0, 1, 0,
			0, 0, 1,
			0, -2, -3,
		}),
		mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 0, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	cm, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}
	dm, err := DiskMargin(sys)
	if err != nil {
		t.Fatal(err)
	}

	if dm.PhaseMargin > cm.PhaseMargin+1 {
		t.Errorf("disk PM %v > classical PM %v (should be ≤)", dm.PhaseMargin, cm.PhaseMargin)
	}
	if !math.IsInf(cm.GainMargin, 1) {
		gmHighDB := dm.GainMarginDB[1]
		if gmHighDB > cm.GainMargin+1 {
			t.Errorf("disk GM_high %v dB > classical GM %v dB", gmHighDB, cm.GainMargin)
		}
	}
}

func TestDiskMargin_MIMOReject(t *testing.T) {
	sys, err := NewGain(mat.NewDense(2, 2, []float64{1, 0, 0, 1}), 0)
	if err != nil {
		t.Fatal(err)
	}

	_, err = DiskMargin(sys)
	if err == nil {
		t.Fatal("expected error for MIMO system")
	}
}

// === Benchmarks ===

func BenchmarkMargin_SISO(b *testing.B) {
	sys, _ := New(
		mat.NewDense(3, 3, []float64{
			0, 1, 0,
			0, 0, 1,
			0, -2, -3,
		}),
		mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 0, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	b.ResetTimer()
	for range b.N {
		Margin(sys)
	}
}

func BenchmarkBandwidth_SISO(b *testing.B) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	b.ResetTimer()
	for range b.N {
		Bandwidth(sys, 0)
	}
}

func BenchmarkDiskMargin_SISO(b *testing.B) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{10}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	b.ResetTimer()
	for range b.N {
		DiskMargin(sys)
	}
}
