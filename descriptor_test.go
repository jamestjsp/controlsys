package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestGeneralizedPoles_Standard(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0, 0, -2})
	E := mat.NewDense(2, 2, []float64{1, 0, 0, 1})

	poles, err := generalizedPoles(A, E, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(poles) != 2 {
		t.Fatalf("got %d poles, want 2", len(poles))
	}

	expected := []complex128{complex(-1, 0), complex(-2, 0)}
	for _, e := range expected {
		found := false
		for _, p := range poles {
			if cmplx.Abs(p-e) < 1e-10 {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected pole %v not found in %v", e, poles)
		}
	}
}

func TestGeneralizedPoles_InfiniteFiltered(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-3, 1, 0, -1})
	E := mat.NewDense(2, 2, []float64{1, 0, 0, 0})

	poles, err := generalizedPoles(A, E, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(poles) != 1 {
		t.Fatalf("got %d finite poles, want 1", len(poles))
	}
	if math.Abs(real(poles[0])-(-3)) > 1e-10 {
		t.Errorf("expected pole -3, got %v", poles[0])
	}
}

func TestDescriptorSystem_Poles(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 2, 0, -3})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	E := mat.NewDense(2, 2, []float64{2, 0, 0, 1})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys.E = E

	if !sys.IsDescriptor() {
		t.Error("expected IsDescriptor() = true")
	}

	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	if len(poles) != 2 {
		t.Fatalf("got %d poles, want 2", len(poles))
	}

	// E^{-1}*A eigenvalues: [2,0;0,1]^{-1}*[-1,2;0,-3] = [-0.5,1;0,-3]
	expected := []complex128{complex(-0.5, 0), complex(-3, 0)}
	for _, e := range expected {
		found := false
		for _, p := range poles {
			if cmplx.Abs(p-e) < 1e-10 {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected pole %v not found in %v", e, poles)
		}
	}
}

func TestDescriptorSystem_Copy(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.E = mat.NewDense(2, 2, []float64{1, 0, 0, 1})

	cp := sys.Copy()
	if cp.E == nil {
		t.Fatal("Copy did not preserve E")
	}
	cp.E.Set(0, 0, 99)
	if sys.E.At(0, 0) == 99 {
		t.Error("Copy shares E backing array")
	}
}

func TestDescriptorSystem_NonDescriptor(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if sys.IsDescriptor() {
		t.Error("standard system should not be descriptor")
	}
}
