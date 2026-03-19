package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestFreqResponse_FirstOrderLowpass(t *testing.T) {
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

	resp, err := sys.FreqResponse([]float64{1.0})
	if err != nil {
		t.Fatal(err)
	}

	mag := cmplx.Abs(resp.At(0, 0, 0))
	want := 1.0 / math.Sqrt(2)
	if math.Abs(mag-want) > 1e-10 {
		t.Fatalf("|H(j)| = %v, want %v", mag, want)
	}
}

func TestFreqResponse_Integrator(t *testing.T) {
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

	freqs := []float64{0.1, 1.0, 10.0}
	resp, err := sys.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}

	for k, w := range freqs {
		mag := cmplx.Abs(resp.At(k, 0, 0))
		if math.Abs(mag-1.0/w) > 1e-10 {
			t.Errorf("w=%v: |H|=%v, want %v", w, mag, 1.0/w)
		}
		ph := cmplx.Phase(resp.At(k, 0, 0)) * 180 / math.Pi
		if math.Abs(ph-(-90)) > 0.1 {
			t.Errorf("w=%v: phase=%v, want -90", w, ph)
		}
	}
}

func TestFreqResponse_Gain(t *testing.T) {
	D := mat.NewDense(1, 1, []float64{3.5})
	sys, err := NewGain(D, 0)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := sys.FreqResponse([]float64{0.1, 1.0, 100.0})
	if err != nil {
		t.Fatal(err)
	}

	for k := 0; k < resp.NFreq; k++ {
		mag := cmplx.Abs(resp.At(k, 0, 0))
		if math.Abs(mag-3.5) > 1e-10 {
			t.Errorf("freq %d: |H|=%v, want 3.5", k, mag)
		}
	}
}

func TestFreqResponse_Discrete(t *testing.T) {
	sysc, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	dt := 0.001
	sysd, err := sysc.Discretize(dt)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{0.01, 0.1, 1.0}
	respC, err := sysc.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}
	respD, err := sysd.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}

	for k, w := range freqs {
		magC := cmplx.Abs(respC.At(k, 0, 0))
		magD := cmplx.Abs(respD.At(k, 0, 0))
		relErr := math.Abs(magC-magD) / magC
		if relErr > 0.01 {
			t.Errorf("w=%v: continuous=%v, discrete=%v, relErr=%v", w, magC, magD, relErr)
		}
	}
}

func TestBode_FirstOrderLowpass(t *testing.T) {
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

	bode, err := sys.Bode([]float64{1.0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	magDB := bode.MagDBAt(0, 0, 0)
	if math.Abs(magDB-(-3.0103)) > 0.01 {
		t.Errorf("mag at w=1: %v dB, want ~-3.01 dB", magDB)
	}

	bode2, err := sys.Bode([]float64{10.0, 100.0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	slope := (bode2.MagDBAt(1, 0, 0) - bode2.MagDBAt(0, 0, 0)) / (math.Log10(100.0) - math.Log10(10.0))
	if math.Abs(slope-(-20)) > 1 {
		t.Errorf("slope = %v dB/decade, want ~-20", slope)
	}
}

func TestBode_AutoFrequency(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-10}),
		mat.NewDense(1, 1, []float64{10}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	bode, err := sys.Bode(nil, 50)
	if err != nil {
		t.Fatal(err)
	}

	if len(bode.Omega) != 50 {
		t.Fatalf("len(omega) = %d, want 50", len(bode.Omega))
	}

	wMin := bode.Omega[0]
	wMax := bode.Omega[len(bode.Omega)-1]
	if wMin > 10 || wMax < 10 {
		t.Errorf("auto range [%v, %v] doesn't cover pole at 10", wMin, wMax)
	}
}

func TestBode_PhaseUnwrap(t *testing.T) {
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{
			0, 1, 0,
			0, 0, 1,
			-1000, -110, -11,
		},
		[]float64{0, 0, 1000},
		[]float64{1, 0, 0},
		[]float64{0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	bode, err := sys.Bode(nil, 500)
	if err != nil {
		t.Fatal(err)
	}

	for k := 1; k < len(bode.Omega); k++ {
		diff := math.Abs(bode.PhaseAt(k, 0, 0) - bode.PhaseAt(k-1, 0, 0))
		if diff > 180 {
			t.Errorf("phase jump at w=%v: %v -> %v (diff=%v)",
				bode.Omega[k], bode.PhaseAt(k-1, 0, 0), bode.PhaseAt(k, 0, 0), diff)
			break
		}
	}
}

func TestBode_MIMO(t *testing.T) {
	sys, err := NewFromSlices(2, 2, 2,
		[]float64{-1, 0, 0, -2},
		[]float64{1, 0, 0, 1},
		[]float64{1, 0, 0, 1},
		[]float64{0, 0, 0, 0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	bode, err := sys.Bode([]float64{1.0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	mag00 := bode.MagDBAt(0, 0, 0)
	mag11 := bode.MagDBAt(0, 1, 1)

	want00 := 20 * math.Log10(1.0/math.Sqrt(2))
	want11 := 20 * math.Log10(1.0/math.Sqrt(5))

	if math.Abs(mag00-want00) > 0.1 {
		t.Errorf("H(0,0) mag = %v dB, want %v", mag00, want00)
	}
	if math.Abs(mag11-want11) > 0.1 {
		t.Errorf("H(1,1) mag = %v dB, want %v", mag11, want11)
	}
}

func TestEvalFr_MatchesFreqResponse(t *testing.T) {
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

	w := 5.0
	evalResult, err := sys.EvalFr(complex(0, w))
	if err != nil {
		t.Fatal(err)
	}

	freqResult, err := sys.FreqResponse([]float64{w})
	if err != nil {
		t.Fatal(err)
	}

	if cmplx.Abs(evalResult[0][0]-freqResult.At(0, 0, 0)) > 1e-15 {
		t.Fatalf("EvalFr=%v != FreqResponse=%v", evalResult[0][0], freqResult.At(0, 0, 0))
	}
}

func TestFreqResponse_Empty(t *testing.T) {
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

	resp, err := sys.FreqResponse([]float64{})
	if err != nil {
		t.Fatal(err)
	}
	if resp != nil {
		t.Fatalf("expected nil, got %v", resp)
	}
}

func TestFreqResponse_InternalDelay_SISO(t *testing.T) {
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

	tau := 0.5
	err = sys.SetInternalDelay(
		[]float64{tau},
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
	)
	if err != nil {
		t.Fatal(err)
	}

	absorbed, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{0.1, 0.5, 1.0, 5.0, 10.0}
	respLFT, err := sys.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}
	respAbs, err := absorbed.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}

	for k, w := range freqs {
		magLFT := cmplx.Abs(respLFT.At(k, 0, 0))
		magAbs := cmplx.Abs(respAbs.At(k, 0, 0))
		relErr := math.Abs(magLFT-magAbs) / math.Max(magAbs, 1e-15)
		if relErr > 0.05 {
			t.Errorf("w=%v: LFT mag=%v, absorbed mag=%v, relErr=%v", w, magLFT, magAbs, relErr)
		}
	}
}

func TestFreqResponse_InternalDelay_D22Zero(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	err = sys.SetInternalDelay(
		[]float64{0.3},
		mat.NewDense(2, 1, []float64{0.5, 1}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
	)
	if err != nil {
		t.Fatal(err)
	}

	absorbed, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{0.1, 1.0, 10.0}
	respLFT, err := sys.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}
	respAbs, err := absorbed.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}

	for k, w := range freqs {
		magLFT := cmplx.Abs(respLFT.At(k, 0, 0))
		magAbs := cmplx.Abs(respAbs.At(k, 0, 0))
		relErr := math.Abs(magLFT-magAbs) / math.Max(magAbs, 1e-15)
		if relErr > 0.05 {
			t.Errorf("w=%v: LFT mag=%v, absorbed mag=%v, relErr=%v", w, magLFT, magAbs, relErr)
		}
	}
}

func TestFreqResponse_InternalDelay_MIMO(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	err = sys.SetInternalDelay(
		[]float64{0.2, 0.5},
		mat.NewDense(2, 2, []float64{0.5, 0, 0, 0.3}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
	)
	if err != nil {
		t.Fatal(err)
	}

	absorbed, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{0.1, 1.0, 5.0}
	respLFT, err := sys.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}
	respAbs, err := absorbed.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}

	for k, w := range freqs {
		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				magLFT := cmplx.Abs(respLFT.At(k, i, j))
				magAbs := cmplx.Abs(respAbs.At(k, i, j))
				absErr := math.Abs(magLFT - magAbs)
				if absErr > 1e-6 && absErr/math.Max(magAbs, 1e-15) > 0.05 {
					t.Errorf("w=%v [%d,%d]: LFT=%v, absorbed=%v, absErr=%v",
						w, i, j, magLFT, magAbs, absErr)
				}
			}
		}
	}
}

func TestFreqResponse_InternalDelay_Discrete(t *testing.T) {
	dt := 1.0
	sys, err := New(
		mat.NewDense(1, 1, []float64{0.9}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		dt,
	)
	if err != nil {
		t.Fatal(err)
	}

	err = sys.SetInternalDelay(
		[]float64{2},
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
	)
	if err != nil {
		t.Fatal(err)
	}

	absorbed, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{0.1, 0.5, 1.0}
	respLFT, err := sys.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}
	respAbs, err := absorbed.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}

	for k, w := range freqs {
		magLFT := cmplx.Abs(respLFT.At(k, 0, 0))
		magAbs := cmplx.Abs(respAbs.At(k, 0, 0))
		relErr := math.Abs(magLFT-magAbs) / math.Max(magAbs, 1e-15)
		if relErr > 1e-10 {
			t.Errorf("w=%v: LFT=%v, absorbed=%v, relErr=%v", w, magLFT, magAbs, relErr)
		}
	}
}

func TestBode_InternalDelay(t *testing.T) {
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

	err = sys.SetInternalDelay(
		[]float64{0.5},
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
	)
	if err != nil {
		t.Fatal(err)
	}

	bode, err := sys.Bode([]float64{0.1, 1.0, 10.0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	if len(bode.Omega) != 3 {
		t.Fatalf("expected 3 freq points, got %d", len(bode.Omega))
	}

	for k := range bode.Omega {
		mag := bode.MagDBAt(k, 0, 0)
		if math.IsNaN(mag) || math.IsInf(mag, 0) {
			t.Errorf("w=%v: got NaN/Inf magnitude", bode.Omega[k])
		}
	}
}

func TestEvalFr_InternalDelay(t *testing.T) {
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

	err = sys.SetInternalDelay(
		[]float64{0.5},
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
	)
	if err != nil {
		t.Fatal(err)
	}

	w := 1.0
	evalRes, err := sys.EvalFr(complex(0, w))
	if err != nil {
		t.Fatal(err)
	}

	freqRes, err := sys.FreqResponse([]float64{w})
	if err != nil {
		t.Fatal(err)
	}

	diff := cmplx.Abs(evalRes[0][0] - freqRes.At(0, 0, 0))
	if diff > 1e-14 {
		t.Errorf("EvalFr=%v != FreqResponse=%v (diff=%v)", evalRes[0][0], freqRes.At(0, 0, 0), diff)
	}
}

func TestFreqResponse_IODelay_InputDelay(t *testing.T) {
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

	tau := 0.5
	err = sys.SetInputDelay([]float64{tau})
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{1.0}
	resp, err := sys.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}

	s := complex(0, 1.0)
	wantNoDelay := 1 / (s + 1)
	want := wantNoDelay * cmplx.Exp(-s*complex(tau, 0))

	got := resp.At(0, 0, 0)
	if cmplx.Abs(got-want) > 1e-10 {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestFreqResponse_IODelay_OutputDelay(t *testing.T) {
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

	tau := 0.3
	err = sys.SetOutputDelay([]float64{tau})
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{2.0}
	resp, err := sys.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}

	s := complex(0, 2.0)
	wantNoDelay := 1 / (s + 1)
	want := wantNoDelay * cmplx.Exp(-s*complex(tau, 0))

	got := resp.At(0, 0, 0)
	if cmplx.Abs(got-want) > 1e-10 {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestBode_SecondOrder(t *testing.T) {
	wn := 10.0
	zeta := 0.1
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -wn * wn, -2 * zeta * wn}),
		mat.NewDense(2, 1, []float64{0, wn * wn}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	omega := logspace(math.Log10(1), math.Log10(100), 500)
	bode, err := sys.Bode(omega, 0)
	if err != nil {
		t.Fatal(err)
	}

	peakDB := math.Inf(-1)
	peakW := 0.0
	for k, w := range bode.Omega {
		if bode.MagDBAt(k, 0, 0) > peakDB {
			peakDB = bode.MagDBAt(k, 0, 0)
			peakW = w
		}
	}

	expectedPeak := wn * math.Sqrt(1-2*zeta*zeta)
	if math.Abs(peakW-expectedPeak)/expectedPeak > 0.05 {
		t.Errorf("resonance at w=%v, want ~%v", peakW, expectedPeak)
	}

	if peakDB < 10 {
		t.Errorf("peak mag = %v dB, expected significant resonance", peakDB)
	}
}

func TestEvalFr_IODelay_Continuous(t *testing.T) {
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
	sys.InputDelay = []float64{2.0}
	
	w := 5.0
	s := complex(0, w)
	val, err := sys.EvalFr(s)
	if err != nil {
		t.Fatal(err)
	}
	
	// Transfer function is 1/(s+1) * e^{-2s}
	tf := 1.0 / (s + 1)
	expected := tf * cmplx.Exp(-s * complex(2.0, 0))
	
	if cmplx.Abs(val[0][0]-expected) > 1e-10 {
		t.Errorf("EvalFr with I/O delay mismatch. got: %v, want: %v", val[0][0], expected)
	}
}

func TestEvalFr_IODelay_Discrete(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.OutputDelay = []float64{3.0}
	
	w := 2.0
	s := cmplx.Exp(complex(0, w*sys.Dt))
	val, err := sys.EvalFr(s)
	if err != nil {
		t.Fatal(err)
	}
	
	// Transfer function is 1/(z-0.5) / z^3
	tf := 1.0 / (s - 0.5)
	expected := tf / (s * s * s)
	
	if cmplx.Abs(val[0][0]-expected) > 1e-10 {
		t.Errorf("EvalFr with I/O delay discrete mismatch. got: %v, want: %v", val[0][0], expected)
	}
}
