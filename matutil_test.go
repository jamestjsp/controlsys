package controlsys

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func assertMatClose(t *testing.T, name string, got, want *mat.Dense, tol float64) {
	t.Helper()
	r1, c1 := got.Dims()
	r2, c2 := want.Dims()
	if r1 != r2 || c1 != c2 {
		t.Fatalf("%s: dim mismatch got %dx%d want %dx%d", name, r1, c1, r2, c2)
	}
	for i := 0; i < r1; i++ {
		for j := 0; j < c1; j++ {
			if diff := math.Abs(got.At(i, j) - want.At(i, j)); diff > tol {
				t.Errorf("%s[%d,%d] = %v, want %v (diff %v)", name, i, j, got.At(i, j), want.At(i, j), diff)
			}
		}
	}
}

func TestDenseCopyTo(t *testing.T) {
	src := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	dst := mat.NewDense(2, 3, nil)

	denseCopyTo(dst, src)
	assertMatClose(t, "copy", dst, src, 0)

	src.Set(0, 0, 99)
	if dst.At(0, 0) == 99 {
		t.Fatal("dst should not alias src backing data")
	}

	denseCopyTo(dst, nil)
	assertMatClose(t, "nil-zero", dst, mat.NewDense(2, 3, nil), 0)
}

func matEqual(a, b *mat.Dense, tol float64) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	ra, ca := a.Dims()
	rb, cb := b.Dims()
	if ra != rb || ca != cb {
		return false
	}
	for i := 0; i < ra; i++ {
		for j := 0; j < ca; j++ {
			if math.Abs(a.At(i, j)-b.At(i, j)) > tol {
				return false
			}
		}
	}
	return true
}
