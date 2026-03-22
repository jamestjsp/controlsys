package controlsys

import (
	"errors"
	"math"
	"testing"
)

func checkDims(t *testing.T, sys *System, wantN, wantM, wantP int) {
	t.Helper()
	n, m, p := sys.Dims()
	if n != wantN || m != wantM || p != wantP {
		t.Fatalf("dims = (%d,%d,%d), want (%d,%d,%d)", n, m, p, wantN, wantM, wantP)
	}
}

func checkD(t *testing.T, sys *System, want [][]float64) {
	t.Helper()
	for i, row := range want {
		for j, v := range row {
			if math.Abs(sys.D.At(i, j)-v) > 1e-14 {
				t.Errorf("D[%d,%d] = %v, want %v", i, j, sys.D.At(i, j), v)
			}
		}
	}
}

func TestSumBlk_Simple(t *testing.T) {
	sys, err := SumBlk("e = r - y")
	if err != nil {
		t.Fatal(err)
	}
	checkDims(t, sys, 0, 2, 1)
	checkD(t, sys, [][]float64{{1, -1}})
}

func TestSumBlk_WithCoefficient(t *testing.T) {
	sys, err := SumBlk("e = 2*r - 0.5*y")
	if err != nil {
		t.Fatal(err)
	}
	checkDims(t, sys, 0, 2, 1)
	checkD(t, sys, [][]float64{{2, -0.5}})
}

func TestSumBlk_UnaryMinus(t *testing.T) {
	sys, err := SumBlk("e = -y")
	if err != nil {
		t.Fatal(err)
	}
	checkDims(t, sys, 0, 1, 1)
	checkD(t, sys, [][]float64{{-1}})
}

func TestSumBlk_ThreeTermSum(t *testing.T) {
	sys, err := SumBlk("e = r - y + d")
	if err != nil {
		t.Fatal(err)
	}
	checkDims(t, sys, 0, 3, 1)
	checkD(t, sys, [][]float64{{1, -1, 1}})
}

func TestSumBlk_Multiple(t *testing.T) {
	sys, err := SumBlk("e = r - y; u = 3*e")
	if err != nil {
		t.Fatal(err)
	}
	checkDims(t, sys, 0, 3, 2)
	checkD(t, sys, [][]float64{
		{1, -1, 0},
		{0, 0, 3},
	})
}

func TestSumBlk_ScalarWidth(t *testing.T) {
	sys, err := SumBlk("e = r - y", 2)
	if err != nil {
		t.Fatal(err)
	}
	checkDims(t, sys, 0, 4, 2)
	checkD(t, sys, [][]float64{
		{1, 0, -1, 0},
		{0, 1, 0, -1},
	})
}

func TestSumBlk_PerSignalWidth(t *testing.T) {
	sys, err := SumBlk("e = r - y", 2, 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	checkDims(t, sys, 0, 4, 2)
	checkD(t, sys, [][]float64{
		{1, 0, 0, -1},
		{0, 1, 0, 0},
	})
}

func TestSumBlk_Empty(t *testing.T) {
	_, err := SumBlk("")
	if !errors.Is(err, ErrInvalidExpression) {
		t.Fatalf("err = %v, want ErrInvalidExpression", err)
	}
}

func TestSumBlk_NoEquals(t *testing.T) {
	_, err := SumBlk("r - y")
	if !errors.Is(err, ErrInvalidExpression) {
		t.Fatalf("err = %v, want ErrInvalidExpression", err)
	}
}

func TestSumBlk_DuplicateOutput(t *testing.T) {
	_, err := SumBlk("e = r; e = y")
	if !errors.Is(err, ErrInvalidExpression) {
		t.Fatalf("err = %v, want ErrInvalidExpression", err)
	}
}

func TestSumBlk_InvalidToken(t *testing.T) {
	_, err := SumBlk("e = r @ y")
	if !errors.Is(err, ErrInvalidExpression) {
		t.Fatalf("err = %v, want ErrInvalidExpression", err)
	}
}

func TestSumBlk_FloatCoeff(t *testing.T) {
	sys, err := SumBlk("e = 2.5*r")
	if err != nil {
		t.Fatal(err)
	}
	checkDims(t, sys, 0, 1, 1)
	checkD(t, sys, [][]float64{{2.5}})
}

func TestSumBlk_WidthMismatch(t *testing.T) {
	_, err := SumBlk("e = r - y", 1, 2)
	if !errors.Is(err, ErrInvalidExpression) {
		t.Fatalf("err = %v, want ErrInvalidExpression", err)
	}
}
