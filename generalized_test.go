package controlsys

import (
	"errors"
	"math/cmplx"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestGeneralizedModelCurrentValueAndAnalysisPoint(t *testing.T) {
	k, _ := NewTunableReal("K", 2, TunableBounds{Lower: 0, Upper: 10})
	controller := NewTunableGain("Kblock", [][]*TunableReal{{k}}, 0)
	gm, err := NewGeneralizedModel("loop", controller)
	if err != nil {
		t.Fatalf("NewGeneralizedModel: %v", err)
	}
	gm.SetInputName("error")
	gm.SetOutputName("actuator")
	gm.InsertAnalysisPoint("plant_input")

	sys, err := gm.CurrentSystem()
	if err != nil {
		t.Fatalf("CurrentSystem: %v", err)
	}
	if sys.D.At(0, 0) != 2 {
		t.Fatalf("current gain = %g, want 2", sys.D.At(0, 0))
	}
	if !sameStrings(sys.InputName, []string{"error"}) || !sameStrings(sys.OutputName, []string{"actuator"}) {
		t.Fatalf("metadata = %v/%v", sys.InputName, sys.OutputName)
	}
	if !gm.HasAnalysisPoint("plant_input") {
		t.Fatal("analysis point not found")
	}
	if _, err := gm.AnalysisPoint("missing"); !errors.Is(err, ErrSignalNotFound) {
		t.Fatalf("missing analysis point err = %v, want ErrSignalNotFound", err)
	}
}

func TestGeneralizedClosedLoopAnalysisHelpers(t *testing.T) {
	plant, err := New(
		mat.NewDense(2, 2, []float64{-1, 0.4, -2, -3}),
		mat.NewDense(2, 1, []float64{1, -0.5}),
		mat.NewDense(1, 2, []float64{2, -1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	k, _ := NewTunableReal("K", 1.5, TunableBounds{Lower: 0.1, Upper: 5})
	controller := NewTunableGain("Kblock", [][]*TunableReal{{k}}, 0)
	loop, err := NewGeneralizedClosedLoop("cl", plant, controller, "u")
	if err != nil {
		t.Fatalf("NewGeneralizedClosedLoop: %v", err)
	}

	L, err := loop.OpenLoop("u")
	if err != nil {
		t.Fatalf("OpenLoop: %v", err)
	}
	T, err := loop.ComplementarySensitivity("u")
	if err != nil {
		t.Fatalf("ComplementarySensitivity: %v", err)
	}
	S, err := loop.Sensitivity("u")
	if err != nil {
		t.Fatalf("Sensitivity: %v", err)
	}
	CL, err := loop.ClosedLoop("u")
	if err != nil {
		t.Fatalf("ClosedLoop: %v", err)
	}
	if _, m, p := L.Dims(); m != 1 || p != 1 {
		t.Fatalf("open-loop dims = (_, %d, %d), want SISO", m, p)
	}
	omega := []float64{0.2, 1.0}
	tResp, _ := T.FreqResponse(omega)
	clResp, _ := CL.FreqResponse(omega)
	for i := range omega {
		if cmplx.Abs(tResp.At(i, 0, 0)-clResp.At(i, 0, 0)) > 1e-10 {
			t.Fatalf("closed-loop and complementary sensitivity differ at %g", omega[i])
		}
	}
	sResp, _ := S.FreqResponse(omega)
	for i := range omega {
		sum := sResp.At(i, 0, 0) + tResp.At(i, 0, 0)
		if cmplx.Abs(sum-1) > 1e-8 {
			t.Fatalf("S+T = %v at %g, want 1", sum, omega[i])
		}
	}
}
