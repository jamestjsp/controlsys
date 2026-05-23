package controlsys

import (
	"errors"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestPassivityDenseAndFRD(t *testing.T) {
	passive := makeSISO(-1, 1, 1, 0)
	result, err := Passive(passive, nil)
	if err != nil {
		t.Fatalf("Passive: %v", err)
	}
	if !result.Passive || result.MinHermitianPart < 0 {
		t.Fatalf("expected passive result, got %#v", result)
	}

	nonpassive := makeSISO(-1, 1, -1, 0)
	bad, err := Passive(nonpassive, nil)
	if err != nil {
		t.Fatalf("Passive nonpassive: %v", err)
	}
	if bad.Passive {
		t.Fatalf("expected nonpassive result, got %#v", bad)
	}

	frd, err := passive.FRD([]float64{0.1, 1, 10})
	if err != nil {
		t.Fatal(err)
	}
	frdResult, err := FRDPassive(frd, nil)
	if err != nil {
		t.Fatalf("FRDPassive: %v", err)
	}
	if !frdResult.Passive {
		t.Fatalf("expected passive FRD result, got %#v", frdResult)
	}
}

func TestSpectralFactorStaticGainAndUnsupportedCases(t *testing.T) {
	sys, err := NewGain(mat.NewDense(1, 1, []float64{4}), 0)
	if err != nil {
		t.Fatal(err)
	}
	factor, err := SpectralFactor(sys)
	if err != nil {
		t.Fatalf("SpectralFactor: %v", err)
	}
	if factor.D.At(0, 0) != 2 {
		t.Fatalf("factor D = %g, want 2", factor.D.At(0, 0))
	}

	dynamic := makeSISO(-1, 1, 1, 0)
	if _, err := SpectralFactor(dynamic); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("dynamic spectral factor err = %v, want ErrDimensionMismatch", err)
	}
	delayed := sys.Copy()
	delayed.InputDelay = []float64{1}
	if _, err := SpectralFactor(delayed); !errors.Is(err, ErrDescriptorUnsupported) {
		t.Fatalf("delayed spectral factor err = %v, want ErrDescriptorUnsupported", err)
	}
}
