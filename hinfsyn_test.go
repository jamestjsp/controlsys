package controlsys

import (
	"errors"
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestHinfSyn_Simple(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 3, []float64{
		1, 0, 0,
		0, 1, 1,
	})
	C := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		1, 0,
	})
	D := mat.NewDense(3, 3, []float64{
		0, 0, 0,
		0, 0, 1,
		0.1, 0.1, 0,
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := HinfSyn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	kn, km, kp := res.K.Dims()
	if kn != 2 {
		t.Errorf("controller states: got %d, want 2", kn)
	}
	if km != 1 {
		t.Errorf("controller inputs: got %d, want 1", km)
	}
	if kp != 1 {
		t.Errorf("controller outputs: got %d, want 1", kp)
	}

	for _, p := range res.CLPoles {
		if real(p) >= 0 {
			t.Errorf("unstable closed-loop pole: %v", p)
		}
	}

	if res.X == nil {
		t.Error("X is nil")
	}
	if res.Y == nil {
		t.Error("Y is nil")
	}
	if res.GammaOpt <= 0 {
		t.Errorf("gamma should be positive, got %v", res.GammaOpt)
	}
}

func TestHinfSyn_DiscreteError(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	D := mat.NewDense(2, 2, nil)

	P, err := New(A, B, C, D, 0.01)
	if err != nil {
		t.Fatal(err)
	}

	_, err = HinfSyn(P, 1, 1)
	if !errors.Is(err, ErrWrongDomain) {
		t.Errorf("got %v, want ErrWrongDomain", err)
	}
}

func TestHinfSyn_DimensionErrors(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 3, []float64{1, 0, 0, 0, 1, 1})
	C := mat.NewDense(3, 2, []float64{1, 0, 0, 0, 1, 0})
	D := mat.NewDense(3, 3, nil)

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name  string
		nmeas int
		ncont int
	}{
		{"nmeas=0", 0, 1},
		{"ncont=0", 1, 0},
		{"nmeas too large", 4, 1},
		{"ncont too large", 1, 4},
		{"nmeas=p", 3, 1},
		{"ncont=m", 1, 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := HinfSyn(P, tt.nmeas, tt.ncont)
			if !errors.Is(err, ErrInvalidPartition) {
				t.Errorf("got %v, want ErrInvalidPartition", err)
			}
		})
	}
}

func TestHinfSyn_GammaPositive(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 3, []float64{
		1, 0, 0,
		0, 1, 1,
	})
	C := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		1, 0,
	})
	D := mat.NewDense(3, 3, []float64{
		0, 0, 0,
		0, 0, 1,
		0.1, 0.1, 0,
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := HinfSyn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	if res.GammaOpt <= 0 {
		t.Errorf("gamma must be positive, got %v", res.GammaOpt)
	}
	if math.IsInf(res.GammaOpt, 0) || math.IsNaN(res.GammaOpt) {
		t.Errorf("gamma must be finite, got %v", res.GammaOpt)
	}
}

func TestHinfSyn_NonNormalizedD12D21(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 3, []float64{
		1, 0, 0,
		0, 1, 1,
	})
	C := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		1, 0,
	})
	// D12 = [0; 2] (not [0; 1]), D21 = [0.3 0.3] (not [0.1 0.1])
	D := mat.NewDense(3, 3, []float64{
		0, 0, 0,
		0, 0, 2,
		0.3, 0.3, 0,
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := HinfSyn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	for _, p := range res.CLPoles {
		if real(p) >= 0 {
			t.Errorf("unstable closed-loop pole: %v", p)
		}
	}
	if res.GammaOpt <= 0 {
		t.Errorf("gamma should be positive, got %v", res.GammaOpt)
	}
	if math.IsInf(res.GammaOpt, 0) || math.IsNaN(res.GammaOpt) {
		t.Errorf("gamma should be finite, got %v", res.GammaOpt)
	}
}

func TestHinfSyn_Stability(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 3, []float64{
		1, 0, 0,
		0, 1, 1,
	})
	C := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 0,
		1, 0,
	})
	D := mat.NewDense(3, 3, []float64{
		0, 0, 0,
		0, 0, 1,
		0.1, 0.1, 0,
	})

	P, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := HinfSyn(P, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	for _, p := range res.CLPoles {
		if real(p) >= -1e-10 {
			t.Errorf("closed-loop pole not in open left half-plane: %v", p)
		}
	}
}
