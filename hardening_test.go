package controlsys

import (
	"errors"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestStateSpace_RejectsImproperTF(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 0}}}, // s (degree 1)
		Den: [][]float64{{1}},         // 1 (degree 0)
	}
	_, err := tf.StateSpace(nil)
	if !errors.Is(err, ErrImproperTF) {
		t.Errorf("expected ErrImproperTF, got %v", err)
	}
}

func TestStateSpace_AcceptsProperTF(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{1}}},
		Den: [][]float64{{1, 1}},
	}
	_, err := tf.StateSpace(nil)
	if err != nil {
		t.Errorf("proper TF should be accepted, got %v", err)
	}
}

func TestStateSpace_AcceptsBiproperTF(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 1}}},
		Den: [][]float64{{1, 2}},
	}
	_, err := tf.StateSpace(nil)
	if err != nil {
		t.Errorf("biproper TF (equal degree) should be accepted, got %v", err)
	}
}

func TestCare_RejectsNegativeDefiniteQ(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{-1, 0, 0, -1})
	R := mat.NewDense(1, 1, []float64{1})

	_, err := Care(A, B, Q, R, nil)
	if !errors.Is(err, ErrNotPSD) {
		t.Errorf("expected ErrNotPSD for negative definite Q, got %v", err)
	}
}

func TestCare_RejectsIndefiniteQ(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, -1})
	R := mat.NewDense(1, 1, []float64{1})

	_, err := Care(A, B, Q, R, nil)
	if !errors.Is(err, ErrNotPSD) {
		t.Errorf("expected ErrNotPSD for indefinite Q, got %v", err)
	}
}

func TestCare_AcceptsPSDQ(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 0}) // PSD but singular
	R := mat.NewDense(1, 1, []float64{1})

	_, err := Care(A, B, Q, R, nil)
	if err != nil {
		t.Errorf("PSD Q should be accepted, got %v", err)
	}
}

func TestDare_RejectsNegativeDefiniteQ(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{-1, 0, 0, -1})
	R := mat.NewDense(1, 1, []float64{1})

	_, err := Dare(A, B, Q, R, nil)
	if !errors.Is(err, ErrNotPSD) {
		t.Errorf("expected ErrNotPSD for negative definite Q, got %v", err)
	}
}

func TestCare_RejectsNonPDR(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{-1})

	_, err := Care(A, B, Q, R, nil)
	if !errors.Is(err, ErrSingularR) {
		t.Errorf("expected ErrSingularR for non-PD R, got %v", err)
	}
}

func TestKalman_RejectsDescriptor(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	sys.E = mat.NewDense(2, 2, []float64{1, 0, 0, 2})

	Qn := mat.NewDense(1, 1, []float64{1})
	Rn := mat.NewDense(1, 1, []float64{1})
	_, err := Kalman(sys, Qn, Rn, nil)
	if !errors.Is(err, ErrDescriptorRiccati) {
		t.Errorf("expected ErrDescriptorRiccati, got %v", err)
	}
}

func TestLqg_RejectsDescriptor(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	sys.E = mat.NewDense(2, 2, []float64{1, 0, 0, 2})

	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})
	Qn := mat.NewDense(1, 1, []float64{1})
	Rn := mat.NewDense(1, 1, []float64{1})
	_, err := Lqg(sys, Q, R, Qn, Rn, nil)
	if !errors.Is(err, ErrDescriptorRiccati) {
		t.Errorf("expected ErrDescriptorRiccati, got %v", err)
	}
}
