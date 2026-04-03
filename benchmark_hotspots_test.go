package controlsys

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// benchSysNonSym builds a stable system with non-symmetric A for benchmark workloads.
// A has sub/superdiagonal coupling so transposition bugs surface.
func benchSysNonSym(n, m, p int) *System {
	A := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		A.Set(i, i, -float64(i+1)*0.3)
		if i > 0 {
			A.Set(i, i-1, 1.0)
		}
		if i < n-1 {
			A.Set(i, i+1, 0.1*float64(i+1))
		}
	}
	B := mat.NewDense(n, m, nil)
	for i := 0; i < n && i < m; i++ {
		B.Set(i, i, 1)
	}
	if n > m {
		for i := m; i < n; i++ {
			B.Set(i, i%m, 0.5)
		}
	}
	C := mat.NewDense(p, n, nil)
	for i := 0; i < p && i < n; i++ {
		C.Set(i, i, 1)
	}
	if p > 1 && n > 1 {
		C.Set(0, 1, 0.3)
	}
	D := mat.NewDense(p, m, nil)
	sys, _ := New(A, B, C, D, 0)
	return sys
}

func benchFRDFromSys(sys *System, nw int) *FRD {
	omega := logspace(-2, 3, nw)
	f, _ := sys.FRD(omega)
	return f
}

// --------------- FRD stack ---------------

func BenchmarkSystemFRD_SISO_200(b *testing.B)   { benchSystemFRD(b, 2, 1, 1, 200) }
func BenchmarkSystemFRD_SISO_2000(b *testing.B)  { benchSystemFRD(b, 2, 1, 1, 2000) }
func BenchmarkSystemFRD_SISO_10000(b *testing.B) { benchSystemFRD(b, 2, 1, 1, 10000) }
func BenchmarkSystemFRD_MIMO_200(b *testing.B)   { benchSystemFRD(b, 10, 3, 4, 200) }
func BenchmarkSystemFRD_MIMO_2000(b *testing.B)  { benchSystemFRD(b, 10, 3, 4, 2000) }
func BenchmarkSystemFRD_MIMO_10000(b *testing.B) { benchSystemFRD(b, 10, 3, 4, 10000) }
func BenchmarkSystemFRD_Large_2000(b *testing.B) { benchSystemFRD(b, 50, 4, 6, 2000) }

func benchSystemFRD(b *testing.B, n, m, p, nw int) {
	sys := benchSysNonSym(n, m, p)
	omega := logspace(-2, 3, nw)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.FRD(omega)
	}
}

func BenchmarkFRDSigma_SISO_200(b *testing.B)   { benchFRDSigma(b, 2, 1, 1, 200) }
func BenchmarkFRDSigma_SISO_2000(b *testing.B)  { benchFRDSigma(b, 2, 1, 1, 2000) }
func BenchmarkFRDSigma_MIMO_200(b *testing.B)   { benchFRDSigma(b, 10, 3, 4, 200) }
func BenchmarkFRDSigma_MIMO_2000(b *testing.B)  { benchFRDSigma(b, 10, 3, 4, 2000) }
func BenchmarkFRDSigma_MIMO_10000(b *testing.B) { benchFRDSigma(b, 10, 3, 4, 10000) }
func BenchmarkFRDSigma_Large_2000(b *testing.B) { benchFRDSigma(b, 50, 4, 6, 2000) }

func benchFRDSigma(b *testing.B, n, m, p, nw int) {
	sys := benchSysNonSym(n, m, p)
	f := benchFRDFromSys(sys, nw)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.Sigma()
	}
}

func BenchmarkFRDMargin_200(b *testing.B)   { benchFRDMargin(b, 4, 200) }
func BenchmarkFRDMargin_2000(b *testing.B)  { benchFRDMargin(b, 4, 2000) }
func BenchmarkFRDMargin_10000(b *testing.B) { benchFRDMargin(b, 4, 10000) }

func BenchmarkFRDMargin_NegGM_2000(b *testing.B) {
	A := mat.NewDense(3, 3, []float64{-1, 1, 0, 0, -1, 1, 0, 0, -1})
	B := mat.NewDense(3, 1, []float64{0, 0, 10})
	C := mat.NewDense(1, 3, []float64{1, 0, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)
	f := benchFRDFromSys(sys, 2000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FRDMargin(f)
	}
}

func benchFRDMargin(b *testing.B, n, nw int) {
	sys := benchSysNonSym(n, 1, 1)
	f := benchFRDFromSys(sys, nw)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FRDMargin(f)
	}
}

func BenchmarkFRDSigma_MIMO5x5_2000(b *testing.B) { benchFRDSigma(b, 20, 5, 5, 2000) }

func BenchmarkFRDSeries_SISO_200(b *testing.B)   { benchFRDSeries(b, 2, 1, 1, 200) }
func BenchmarkFRDSeries_SISO_2000(b *testing.B)  { benchFRDSeries(b, 2, 1, 1, 2000) }
func BenchmarkFRDSeries_MIMO_200(b *testing.B)   { benchFRDSeries(b, 10, 3, 3, 200) }
func BenchmarkFRDSeries_MIMO_2000(b *testing.B)  { benchFRDSeries(b, 10, 3, 3, 2000) }
func BenchmarkFRDSeries_MIMO_10000(b *testing.B) { benchFRDSeries(b, 10, 3, 3, 10000) }

func benchFRDSeries(b *testing.B, n, m, p, nw int) {
	s1 := benchSysNonSym(n, m, p)
	s2 := benchSysNonSym(n, p, m)
	f1 := benchFRDFromSys(s1, nw)
	f2 := benchFRDFromSys(s2, nw)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FRDSeries(f1, f2)
	}
}

func BenchmarkFRDParallel_SISO_2000(b *testing.B) { benchFRDParallel(b, 2, 1, 1, 2000) }
func BenchmarkFRDParallel_MIMO_2000(b *testing.B) { benchFRDParallel(b, 10, 3, 3, 2000) }

func benchFRDParallel(b *testing.B, n, m, p, nw int) {
	sys := benchSysNonSym(n, m, p)
	f1 := benchFRDFromSys(sys, nw)
	f2 := benchFRDFromSys(sys, nw)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FRDParallel(f1, f2)
	}
}

func BenchmarkFRDFeedback_SISO_200(b *testing.B)   { benchFRDFeedback(b, 2, 1, 1, 200) }
func BenchmarkFRDFeedback_SISO_2000(b *testing.B)  { benchFRDFeedback(b, 2, 1, 1, 2000) }
func BenchmarkFRDFeedback_MIMO_200(b *testing.B)   { benchFRDFeedback(b, 10, 3, 3, 200) }
func BenchmarkFRDFeedback_MIMO_2000(b *testing.B)  { benchFRDFeedback(b, 10, 3, 3, 2000) }
func BenchmarkFRDFeedback_MIMO_10000(b *testing.B) { benchFRDFeedback(b, 10, 3, 3, 10000) }

func benchFRDFeedback(b *testing.B, n, m, p, nw int) {
	plant := benchSysNonSym(n, m, p)
	ctrl := benchSysNonSym(n/2+1, p, m)
	fp := benchFRDFromSys(plant, nw)
	fc := benchFRDFromSys(ctrl, nw)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FRDFeedback(fp, fc, -1)
	}
}

// --------------- Time-response wrappers ---------------

func BenchmarkStep_SISO_N2(b *testing.B)  { benchStep(b, 2, 1, 1) }
func BenchmarkStep_SISO_N10(b *testing.B) { benchStep(b, 10, 1, 1) }
func BenchmarkStep_SISO_N50(b *testing.B) { benchStep(b, 50, 1, 1) }
func BenchmarkStep_MIMO_N10(b *testing.B) { benchStep(b, 10, 3, 4) }

func benchStep(b *testing.B, n, m, p int) {
	sys := benchSysNonSym(n, m, p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Step(sys, 10)
	}
}

func BenchmarkImpulse_SISO_N2(b *testing.B)  { benchImpulse(b, 2, 1, 1) }
func BenchmarkImpulse_SISO_N10(b *testing.B) { benchImpulse(b, 10, 1, 1) }
func BenchmarkImpulse_SISO_N50(b *testing.B) { benchImpulse(b, 50, 1, 1) }
func BenchmarkImpulse_MIMO_N10(b *testing.B) { benchImpulse(b, 10, 3, 4) }

func benchImpulse(b *testing.B, n, m, p int) {
	sys := benchSysNonSym(n, m, p)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Impulse(sys, 10)
	}
}

func BenchmarkLsim_SISO_N2(b *testing.B)  { benchLsimB(b, 2, 1, 1, 500) }
func BenchmarkLsim_SISO_N10(b *testing.B) { benchLsimB(b, 10, 1, 1, 500) }
func BenchmarkLsim_SISO_N50(b *testing.B) { benchLsimB(b, 50, 1, 1, 500) }
func BenchmarkLsim_MIMO_N10(b *testing.B) { benchLsimB(b, 10, 3, 4, 500) }
func BenchmarkLsim_SISO_1e4(b *testing.B) { benchLsimB(b, 4, 1, 1, 10000) }
func BenchmarkLsim_MIMO_1e4(b *testing.B) { benchLsimB(b, 10, 4, 4, 10000) }

func BenchmarkLsim_Discrete_SISO(b *testing.B) {
	sys := benchSysNonSym(4, 1, 1)
	dsys, _ := sys.DiscretizeZOH(0.01)
	steps := 1000
	t := make([]float64, steps)
	for k := range t {
		t[k] = float64(k) * 0.01
	}
	u := mat.NewDense(steps, 1, nil)
	for k := range steps {
		u.Set(k, 0, 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Lsim(dsys, u, t, nil)
	}
}

func benchLsimB(b *testing.B, n, m, p, steps int) {
	sys := benchSysNonSym(n, m, p)
	dt := 0.01
	t := make([]float64, steps)
	for k := range t {
		t[k] = float64(k) * dt
	}
	u := mat.NewDense(steps, m, nil)
	for k := 0; k < steps; k++ {
		for j := 0; j < m; j++ {
			u.Set(k, j, math.Sin(float64(k)*dt*float64(j+1)))
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Lsim(sys, u, t, nil)
	}
}

// --------------- Loopsens ---------------

func BenchmarkLoopsens_SISO_N2(b *testing.B)  { benchLoopsens(b, 2, 1, 1) }
func BenchmarkLoopsens_SISO_N10(b *testing.B) { benchLoopsens(b, 10, 1, 1) }
func BenchmarkLoopsens_MIMO_N10(b *testing.B) { benchLoopsens(b, 10, 3, 3) }

func benchLoopsens(b *testing.B, n, m, p int) {
	P := benchSysNonSym(n, m, p)
	C := benchSysNonSym(n/2+1, p, m)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Loopsens(P, C)
	}
}

// --------------- Transform / decomposition ---------------

func BenchmarkCanon_Modal_N2(b *testing.B)   { benchCanon(b, CanonModal, 2) }
func BenchmarkCanon_Modal_N10(b *testing.B)  { benchCanon(b, CanonModal, 10) }
func BenchmarkCanon_Modal_N50(b *testing.B)  { benchCanon(b, CanonModal, 50) }
func BenchmarkCanon_Modal_N100(b *testing.B) { benchCanon(b, CanonModal, 100) }
func BenchmarkCanon_Comp_N2(b *testing.B)    { benchCanon(b, CanonCompanion, 2) }
func BenchmarkCanon_Comp_N10(b *testing.B)   { benchCanon(b, CanonCompanion, 10) }

func benchCanon(b *testing.B, form CanonForm, n int) {
	sys := benchSysNonSym(n, 1, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Canon(sys, form)
	}
}

func BenchmarkStabsep_N2(b *testing.B)   { benchStabsep(b, 2) }
func BenchmarkStabsep_N10(b *testing.B)  { benchStabsep(b, 10) }
func BenchmarkStabsep_N50(b *testing.B)  { benchStabsep(b, 50) }
func BenchmarkStabsep_N100(b *testing.B) { benchStabsep(b, 100) }

func benchStabsep(b *testing.B, n int) {
	A := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		if i%2 == 0 {
			A.Set(i, i, -float64(i+1)*0.3)
		} else {
			A.Set(i, i, float64(i)*0.2)
		}
		if i > 0 {
			A.Set(i, i-1, 0.5)
		}
		if i < n-1 {
			A.Set(i, i+1, 0.1)
		}
	}
	B := mat.NewDense(n, 2, nil)
	B.Set(0, 0, 1)
	if n > 1 {
		B.Set(1, 1, 1)
	}
	C := mat.NewDense(2, n, nil)
	C.Set(0, 0, 1)
	if n > 1 {
		C.Set(1, 1, 1)
	}
	D := mat.NewDense(2, 2, nil)
	sys, _ := New(A, B, C, D, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Stabsep(sys)
	}
}

func BenchmarkModsep_N10(b *testing.B) { benchModsep(b, 10) }
func BenchmarkModsep_N50(b *testing.B) { benchModsep(b, 50) }

func benchModsep(b *testing.B, n int) {
	sys := benchSysNonSym(n, 2, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Modsep(sys, 1.0)
	}
}

func BenchmarkSsbal_N2(b *testing.B)  { benchSsbalB(b, 2) }
func BenchmarkSsbal_N10(b *testing.B) { benchSsbalB(b, 10) }
func BenchmarkSsbal_N50(b *testing.B) { benchSsbalB(b, 50) }

func benchSsbalB(b *testing.B, n int) {
	sys := benchSysNonSym(n, 2, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Ssbal(sys)
	}
}

func BenchmarkPrescale_N2(b *testing.B)  { benchPrescaleB(b, 2) }
func BenchmarkPrescale_N10(b *testing.B) { benchPrescaleB(b, 10) }
func BenchmarkPrescale_N50(b *testing.B) { benchPrescaleB(b, 50) }

func benchPrescaleB(b *testing.B, n int) {
	sys := benchSysNonSym(n, 2, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Prescale(sys)
	}
}

func BenchmarkSminreal_N2(b *testing.B)  { benchSminrealB(b, 2) }
func BenchmarkSminreal_N10(b *testing.B) { benchSminrealB(b, 10) }
func BenchmarkSminreal_N50(b *testing.B) { benchSminrealB(b, 50) }

func benchSminrealB(b *testing.B, n int) {
	sys := benchSysNonSym(n, 2, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Sminreal(sys)
	}
}

func BenchmarkInv_SISO_N2(b *testing.B)  { benchInvB(b, 2, 1, 1) }
func BenchmarkInv_SISO_N10(b *testing.B) { benchInvB(b, 10, 1, 1) }
func BenchmarkInv_MIMO_N10(b *testing.B) { benchInvB(b, 10, 3, 3) }

func benchInvB(b *testing.B, n, m, p int) {
	sys := benchSysNonSym(n, m, p)
	sys.D.Set(0, 0, 1)
	for i := 1; i < m && i < p; i++ {
		sys.D.Set(i, i, 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Inv(sys)
	}
}

func BenchmarkCovar_SISO_N2(b *testing.B)   { benchCovarB(b, 2, 1, 1) }
func BenchmarkCovar_SISO_N10(b *testing.B)  { benchCovarB(b, 10, 1, 1) }
func BenchmarkCovar_SISO_N50(b *testing.B)  { benchCovarB(b, 50, 1, 1) }
func BenchmarkCovar_SISO_N100(b *testing.B) { benchCovarB(b, 100, 1, 1) }
func BenchmarkCovar_MIMO_N10(b *testing.B)  { benchCovarB(b, 10, 3, 4) }

func benchCovarB(b *testing.B, n, m, p int) {
	sys := benchSysNonSym(n, m, p)
	W := mat.NewDense(m, m, nil)
	for i := 0; i < m; i++ {
		W.Set(i, i, 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Covar(sys, W)
	}
}

// --------------- Pidtune ---------------

func BenchmarkPidtune_PI(b *testing.B) {
	sys := benchSysNonSym(4, 1, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Pidtune(sys, "PI")
	}
}

func BenchmarkPidtune_PID(b *testing.B) {
	sys := benchSysNonSym(4, 1, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Pidtune(sys, "PID")
	}
}

func BenchmarkPidtune_PIDF(b *testing.B) {
	sys := benchSysNonSym(4, 1, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Pidtune(sys, "PIDF")
	}
}
