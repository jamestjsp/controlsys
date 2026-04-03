package controlsys

import (
	"math/cmplx"
	"testing"
)

func TestRss_Basic(t *testing.T) {
	sys, err := Rss(5, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := sys.Dims()
	if n != 5 || m != 3 || p != 2 {
		t.Fatalf("dims = (%d,%d,%d), want (5,3,2)", n, m, p)
	}

	if !sys.IsContinuous() {
		t.Error("expected continuous system")
	}

	stable, err := sys.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		poles, _ := sys.Poles()
		t.Errorf("system not stable, poles = %v", poles)
	}
}

func TestRss_StrictStability(t *testing.T) {
	for i := 0; i < 20; i++ {
		sys, err := Rss(8, 1, 1)
		if err != nil {
			t.Fatal(err)
		}
		poles, _ := sys.Poles()
		for j, p := range poles {
			if real(p) >= 0 {
				t.Errorf("trial %d: pole[%d] = %v has non-negative real part", i, j, p)
			}
		}
	}
}

func TestRss_ZeroStates(t *testing.T) {
	sys, err := Rss(0, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := sys.Dims()
	if n != 0 || m != 3 || p != 2 {
		t.Fatalf("dims = (%d,%d,%d), want (0,3,2)", n, m, p)
	}
}

func TestDrss_Basic(t *testing.T) {
	sys, err := Drss(4, 1, 1, 0.1)
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := sys.Dims()
	if n != 4 || m != 1 || p != 1 {
		t.Fatalf("dims = (%d,%d,%d), want (4,1,1)", n, m, p)
	}

	if !sys.IsDiscrete() || sys.Dt != 0.1 {
		t.Errorf("expected discrete with dt=0.1, got dt=%g", sys.Dt)
	}

	stable, err := sys.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Error("discrete system not stable")
	}
}

func TestDrss_StrictStability(t *testing.T) {
	for i := 0; i < 20; i++ {
		sys, err := Drss(6, 2, 2, 0.05)
		if err != nil {
			t.Fatal(err)
		}
		poles, _ := sys.Poles()
		for j, p := range poles {
			if cmplx.Abs(p) >= 1 {
				t.Errorf("trial %d: pole[%d] = %v has |p| >= 1", i, j, p)
			}
		}
	}
}

func TestDrss_ZeroStates(t *testing.T) {
	sys, err := Drss(0, 1, 1, 0.5)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := sys.Dims()
	if n != 0 {
		t.Fatalf("n = %d, want 0", n)
	}
}

func TestDrss_InvalidDt(t *testing.T) {
	_, err := Drss(4, 1, 1, 0)
	if err == nil {
		t.Error("expected error for dt=0")
	}

	_, err = Drss(4, 1, 1, -1)
	if err == nil {
		t.Error("expected error for dt<0")
	}
}

func TestRss_InvalidDims(t *testing.T) {
	_, err := Rss(-1, 1, 1)
	if err == nil {
		t.Error("expected error for n<0")
	}

	_, err = Rss(1, 0, 1)
	if err == nil {
		t.Error("expected error for p=0")
	}

	_, err = Rss(1, 1, 0)
	if err == nil {
		t.Error("expected error for m=0")
	}
}
