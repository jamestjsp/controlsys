package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestFRD_SISOFromSystem(t *testing.T) {
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

	omega := []float64{0.1, 1, 10}
	frd, err := sys.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}

	p, m := frd.Dims()
	if p != 1 || m != 1 {
		t.Fatalf("Dims() = (%d,%d), want (1,1)", p, m)
	}
	if frd.NumFrequencies() != 3 {
		t.Fatalf("NumFrequencies() = %d, want 3", frd.NumFrequencies())
	}
	if !frd.IsContinuous() {
		t.Fatal("expected continuous")
	}

	wantH := []complex128{
		complex(1, 0) / complex(1, 0.1),
		complex(1, 0) / complex(1, 1),
		complex(1, 0) / complex(1, 10),
	}
	wantMag := []float64{
		1.0 / math.Sqrt(1.01),
		1.0 / math.Sqrt(2),
		1.0 / math.Sqrt(101),
	}

	for k := range omega {
		h := frd.At(k, 0, 0)
		if cmplx.Abs(h-wantH[k]) > 1e-10 {
			t.Errorf("w=%v: H=%v, want %v", omega[k], h, wantH[k])
		}
		mag := cmplx.Abs(h)
		if math.Abs(mag-wantMag[k]) > 1e-10 {
			t.Errorf("w=%v: |H|=%v, want %v", omega[k], mag, wantMag[k])
		}
	}
}

func TestFRD_MIMOFromSystem(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	omega := []float64{1.0}
	frd, err := sys.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}

	p, m := frd.Dims()
	if p != 2 || m != 2 {
		t.Fatalf("Dims() = (%d,%d), want (2,2)", p, m)
	}

	// H(j) = (jI - A)^{-1} since C=I, B=I, D=0
	// jI - A = [j, -1; 2, j+3]
	// det = j(j+3) + 2 = -1+3j+2 = 1+3j
	det := complex(1, 3)
	want := [2][2]complex128{
		{complex(0, 1) + 3, 1},
		{-2, complex(0, 1)},
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			want[i][j] /= det
		}
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			got := frd.At(0, i, j)
			if cmplx.Abs(got-want[i][j]) > 1e-10 {
				t.Errorf("H[%d][%d] = %v, want %v", i, j, got, want[i][j])
			}
		}
	}
}

func TestFRD_DirectConstruction(t *testing.T) {
	resp := [][][]complex128{
		{{0.5 - 0.5i}},
		{{0.01 - 0.1i}},
	}
	omega := []float64{1, 10}
	frd, err := NewFRD(resp, omega, 0)
	if err != nil {
		t.Fatal(err)
	}

	p, m := frd.Dims()
	if p != 1 || m != 1 {
		t.Fatalf("Dims() = (%d,%d), want (1,1)", p, m)
	}
	if frd.NumFrequencies() != 2 {
		t.Fatalf("NumFrequencies() = %d, want 2", frd.NumFrequencies())
	}

	if frd.At(0, 0, 0) != 0.5-0.5i {
		t.Errorf("At(0,0,0) = %v, want 0.5-0.5i", frd.At(0, 0, 0))
	}
	if frd.At(1, 0, 0) != 0.01-0.1i {
		t.Errorf("At(1,0,0) = %v, want 0.01-0.1i", frd.At(1, 0, 0))
	}
}

func TestFRD_Discrete(t *testing.T) {
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

	omega := []float64{0.1, 1, 5}
	frd, err := sys.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}

	if !frd.IsDiscrete() {
		t.Fatal("expected discrete")
	}
	if frd.IsContinuous() {
		t.Fatal("expected not continuous")
	}

	for k, w := range omega {
		z := cmplx.Exp(complex(0, w*0.1))
		want := 1 / (z - 0.5)
		got := frd.At(k, 0, 0)
		if cmplx.Abs(got-want) > 1e-10 {
			t.Errorf("w=%v: H=%v, want %v", w, got, want)
		}
	}
}

func TestFRD_ValidationErrors(t *testing.T) {
	t.Run("unsorted omega", func(t *testing.T) {
		resp := [][][]complex128{{{1 + 0i}}, {{2 + 0i}}}
		_, err := NewFRD(resp, []float64{10, 1}, 0)
		if err == nil {
			t.Fatal("expected error for unsorted omega")
		}
	})

	t.Run("negative omega", func(t *testing.T) {
		resp := [][][]complex128{{{1 + 0i}}, {{2 + 0i}}}
		_, err := NewFRD(resp, []float64{-1, 1}, 0)
		if err == nil {
			t.Fatal("expected error for negative omega")
		}
	})

	t.Run("mismatched response dimensions", func(t *testing.T) {
		resp := [][][]complex128{
			{{1 + 0i}},
			{{2 + 0i, 3 + 0i}},
		}
		_, err := NewFRD(resp, []float64{1, 2}, 0)
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Fatalf("expected ErrDimensionMismatch, got %v", err)
		}
	})

	t.Run("response/omega length mismatch", func(t *testing.T) {
		resp := [][][]complex128{{{1 + 0i}}}
		_, err := NewFRD(resp, []float64{1, 2}, 0)
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Fatalf("expected ErrDimensionMismatch, got %v", err)
		}
	})

	t.Run("negative sample time", func(t *testing.T) {
		resp := [][][]complex128{{{1 + 0i}}}
		_, err := NewFRD(resp, []float64{1}, -0.1)
		if !errors.Is(err, ErrInvalidSampleTime) {
			t.Fatalf("expected ErrInvalidSampleTime, got %v", err)
		}
	})

	t.Run("exceeds Nyquist", func(t *testing.T) {
		resp := [][][]complex128{{{1 + 0i}}}
		dt := 0.1
		nyq := math.Pi / dt
		_, err := NewFRD(resp, []float64{nyq + 1}, dt)
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Fatalf("expected ErrDimensionMismatch, got %v", err)
		}
	})
}

func TestFRD_CrossValidateWithFreqResponse(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	omega := []float64{0.01, 0.1, 1, 10, 100}
	resp, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	frd, err := sys.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}

	_, m, p := sys.Dims()
	for k := range omega {
		for i := 0; i < p; i++ {
			for j := 0; j < m; j++ {
				got := frd.At(k, i, j)
				want := resp.At(k, i, j)
				if cmplx.Abs(got-want) > 1e-10 {
					t.Errorf("w=%v [%d,%d]: FRD=%v, FreqResponse=%v", omega[k], i, j, got, want)
				}
			}
		}
	}
}

func TestFRD_CrossValidateMIMO(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	omega := []float64{0.01, 0.1, 1, 10, 100}
	resp, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	frd, err := sys.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}

	_, m, p := sys.Dims()
	for k := range omega {
		for i := 0; i < p; i++ {
			for j := 0; j < m; j++ {
				got := frd.At(k, i, j)
				want := resp.At(k, i, j)
				if cmplx.Abs(got-want) > 1e-10 {
					t.Errorf("w=%v [%d,%d]: FRD=%v, FreqResponse=%v", omega[k], i, j, got, want)
				}
			}
		}
	}
}

func TestFRD_EmptyOmega(t *testing.T) {
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

	frd, err := sys.FRD(nil)
	if err != nil {
		t.Fatal(err)
	}
	if frd.NumFrequencies() != 0 {
		t.Errorf("expected 0 frequencies, got %d", frd.NumFrequencies())
	}
}

func TestFRD_EvalFr(t *testing.T) {
	resp := [][][]complex128{
		{{1 + 2i, 3 + 4i}, {5 + 6i, 7 + 8i}},
	}
	frd, err := NewFRD(resp, []float64{1.0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	grid := frd.EvalFr(0)
	if len(grid) != 2 || len(grid[0]) != 2 {
		t.Fatalf("EvalFr dims = %dx%d, want 2x2", len(grid), len(grid[0]))
	}
	if grid[0][0] != 1+2i || grid[0][1] != 3+4i || grid[1][0] != 5+6i || grid[1][1] != 7+8i {
		t.Errorf("EvalFr mismatch: %v", grid)
	}
}

func TestFRD_FreqResponseMatrix(t *testing.T) {
	resp := [][][]complex128{
		{{1 + 2i}},
		{{3 + 4i}},
	}
	frd, err := NewFRD(resp, []float64{1, 10}, 0)
	if err != nil {
		t.Fatal(err)
	}

	frm := frd.FreqResponse()
	if frm.NFreq != 2 || frm.P != 1 || frm.M != 1 {
		t.Fatalf("FreqResponseMatrix dims: NFreq=%d P=%d M=%d", frm.NFreq, frm.P, frm.M)
	}
	if frm.At(0, 0, 0) != 1+2i {
		t.Errorf("At(0,0,0) = %v, want 1+2i", frm.At(0, 0, 0))
	}
	if frm.At(1, 0, 0) != 3+4i {
		t.Errorf("At(1,0,0) = %v, want 3+4i", frm.At(1, 0, 0))
	}
}

func TestFRD_Bode(t *testing.T) {
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

	omega := []float64{0.1, 1, 10}
	frd, err := sys.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}

	bode := frd.Bode()
	wantMagDB := []float64{
		20 * math.Log10(1.0/math.Sqrt(1.01)),
		20 * math.Log10(1.0/math.Sqrt(2)),
		20 * math.Log10(1.0/math.Sqrt(101)),
	}
	for k := range omega {
		got := bode.MagDBAt(k, 0, 0)
		if math.Abs(got-wantMagDB[k]) > 1e-8 {
			t.Errorf("w=%v: magDB=%v, want %v", omega[k], got, wantMagDB[k])
		}
	}
}

func TestFRD_DataIsolation(t *testing.T) {
	resp := [][][]complex128{{{1 + 0i}}}
	omega := []float64{1.0}
	frd, err := NewFRD(resp, omega, 0)
	if err != nil {
		t.Fatal(err)
	}

	resp[0][0][0] = 999 + 0i
	omega[0] = 999
	if frd.At(0, 0, 0) != 1+0i {
		t.Error("FRD response mutated by external modification")
	}
	if frd.Omega[0] != 1.0 {
		t.Error("FRD omega mutated by external modification")
	}
}

func TestFRD_NyquistBoundary(t *testing.T) {
	dt := 0.1
	nyq := math.Pi / dt
	resp := [][][]complex128{{{1 + 0i}}}
	_, err := NewFRD(resp, []float64{nyq}, dt)
	if err != nil {
		t.Fatalf("Nyquist frequency should be allowed, got %v", err)
	}
}

func TestFRD_NyquistFromFRD(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	w := logspace(-1, 1, 50)
	f, err := sys.FRD(w)
	if err != nil {
		t.Fatal(err)
	}
	res, err := f.Nyquist()
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Contour) != 50 {
		t.Fatalf("contour len = %d, want 50", len(res.Contour))
	}
	if cmplx.Abs(res.Contour[0]-f.Response[0][0][0]) > 1e-12 {
		t.Errorf("contour[0] mismatch")
	}
}

func TestFRD_Sigma(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	w := logspace(-1, 1, 20)
	f, err := sys.FRD(w)
	if err != nil {
		t.Fatal(err)
	}
	sig, err := f.Sigma()
	if err != nil {
		t.Fatal(err)
	}
	for k, wk := range w {
		want := 1 / math.Sqrt(1+wk*wk)
		got := sig.sv[k]
		if math.Abs(got-want)/want > 1e-4 {
			t.Errorf("w=%g: sigma=%g, want %g", wk, got, want)
		}
	}
}

func TestFRDMargin(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{0, 1, 0, 0, 0, 1, -6, -11, -6})
	B := mat.NewDense(3, 1, []float64{0, 0, 60})
	C := mat.NewDense(1, 3, []float64{1, 0, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)
	w := logspace(-2, 2, 2000)
	f, err := sys.FRD(w)
	if err != nil {
		t.Fatal(err)
	}
	mr, err := FRDMargin(f)
	if err != nil {
		t.Fatal(err)
	}
	sysMr, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsInf(sysMr.PhaseMargin, 0) {
		t.Skip("system has no gain crossover")
	}
	if math.Abs(mr.PhaseMargin-sysMr.PhaseMargin) > 3 {
		t.Errorf("PM = %g, sys PM = %g", mr.PhaseMargin, sysMr.PhaseMargin)
	}
}

func TestFRD_Sigma_MIMO_Regression(t *testing.T) {
	// Bug: real block form doubles SVs; diag(2,1) was returning [2,2] not [2,1]
	resp := [][][]complex128{
		{{2, 0}, {0, 1}},
	}
	f, _ := NewFRD(resp, []float64{1.0}, 0)
	sig, err := f.Sigma()
	if err != nil {
		t.Fatal(err)
	}
	if sig.nSV != 2 {
		t.Fatalf("nSV = %d, want 2", sig.nSV)
	}
	if math.Abs(sig.sv[0]-2) > 1e-10 {
		t.Errorf("sv[0] = %g, want 2", sig.sv[0])
	}
	if math.Abs(sig.sv[1]-1) > 1e-10 {
		t.Errorf("sv[1] = %g, want 1", sig.sv[1])
	}
}

func TestFRDMargin_NegativeMargins(t *testing.T) {
	// K/(s+1)^3 with K=10: unstable closed-loop, negative margins
	// Margin() returns GM≈-1.94dB, PM≈-7°
	A := mat.NewDense(3, 3, []float64{-1, 1, 0, 0, -1, 1, 0, 0, -1})
	B := mat.NewDense(3, 1, []float64{0, 0, 10})
	C := mat.NewDense(1, 3, []float64{1, 0, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)

	sysMr, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}
	if sysMr.GainMargin >= 0 {
		t.Skipf("expected negative GM, got %g", sysMr.GainMargin)
	}

	w := logspace(-2, 2, 5000)
	f, err := sys.FRD(w)
	if err != nil {
		t.Fatal(err)
	}
	mr, err := FRDMargin(f)
	if err != nil {
		t.Fatal(err)
	}

	if math.IsInf(mr.GainMargin, 1) || math.IsNaN(mr.GainMargin) {
		t.Errorf("FRDMargin GM = %g, want finite negative (sys GM = %g)", mr.GainMargin, sysMr.GainMargin)
	}
	if math.Abs(mr.GainMargin-sysMr.GainMargin) > 1 {
		t.Errorf("FRDMargin GM = %g, sys GM = %g", mr.GainMargin, sysMr.GainMargin)
	}
	if math.IsInf(mr.PhaseMargin, 1) || math.IsNaN(mr.PhaseMargin) {
		t.Errorf("FRDMargin PM = %g, want finite negative (sys PM = %g)", mr.PhaseMargin, sysMr.PhaseMargin)
	}
	if math.Abs(mr.PhaseMargin-sysMr.PhaseMargin) > 3 {
		t.Errorf("FRDMargin PM = %g, sys PM = %g", mr.PhaseMargin, sysMr.PhaseMargin)
	}
	if math.IsNaN(mr.WgFreq) {
		t.Errorf("FRDMargin WgFreq = NaN, want finite (sys = %g)", sysMr.WgFreq)
	}
	if math.IsNaN(mr.WpFreq) {
		t.Errorf("FRDMargin WpFreq = NaN, want finite (sys = %g)", sysMr.WpFreq)
	}
}

func TestLsim_NonUniformGrid_Rejected(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	tNonUniform := []float64{0, 0.1, 0.3, 0.6}
	u := mat.NewDense(4, 1, []float64{1, 1, 1, 1})
	_, err := Lsim(sys, u, tNonUniform, nil)
	if err == nil {
		t.Error("Lsim should reject non-uniform time grid")
	}
}

func TestLsim_DiscreteDtMismatch_Rejected(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.9}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0.1)
	tGrid := []float64{0, 0.05, 0.10, 0.15}
	u := mat.NewDense(4, 1, []float64{1, 1, 1, 1})
	_, err := Lsim(sys, u, tGrid, nil)
	if err == nil {
		t.Error("Lsim should reject discrete system when t spacing != sys.Dt")
	}
}

func TestInv_DelayRejected(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), 0)
	sys.SetInputDelay([]float64{0.5})
	_, err := Inv(sys)
	if err == nil {
		t.Error("Inv should reject delayed system")
	}
}

