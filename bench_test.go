package controlsys

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func BenchmarkSimulateWithDelay_SISO(b *testing.B) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(1, 2, []float64{1, 1})
	D := mat.NewDense(1, 1, []float64{0.5})
	delay := mat.NewDense(1, 1, []float64{5})
	sys, _ := NewWithDelay(A, B, C, D, delay, 1.0)

	steps := 100
	u := mat.NewDense(1, steps, nil)
	for k := 0; k < steps; k++ {
		u.Set(0, k, 1)
	}
	x0 := mat.NewVecDense(2, []float64{1, -0.5})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Simulate(u, x0, nil)
	}
}

func BenchmarkSimulateWithDelay_MIMO(b *testing.B) {
	n, m, p := 10, 4, 6
	A := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		A.Set(i, i, 0.9-float64(i)*0.05)
		if i > 0 {
			A.Set(i, i-1, 0.1)
		}
	}
	B := mat.NewDense(n, m, nil)
	for i := 0; i < n && i < m; i++ {
		B.Set(i, i, 1)
	}
	C := mat.NewDense(p, n, nil)
	for i := 0; i < p && i < n; i++ {
		C.Set(i, i, 1)
	}
	D := mat.NewDense(p, m, nil)

	delay := mat.NewDense(p, m, nil)
	for i := 0; i < p; i++ {
		for j := 0; j < m; j++ {
			delay.Set(i, j, float64(2+i+j))
		}
	}
	sys, _ := NewWithDelay(A, B, C, D, delay, 1.0)

	steps := 200
	u := mat.NewDense(m, steps, nil)
	for j := 0; j < m; j++ {
		for k := 0; k < steps; k++ {
			u.Set(j, k, math.Sin(float64(k)*0.1+float64(j)))
		}
	}
	x0 := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		x0.SetVec(i, float64(i+1)*0.1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Simulate(u, x0, nil)
	}
}

func BenchmarkBilinearDiscretize(b *testing.B) {
	n := 20
	A := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		A.Set(i, i, -float64(i+1)*0.5)
		if i > 0 {
			A.Set(i, i-1, 1)
		}
	}
	B := mat.NewDense(n, 3, nil)
	for i := 0; i < 3; i++ {
		B.Set(i, i, 1)
	}
	C := mat.NewDense(2, n, nil)
	C.Set(0, 0, 1)
	C.Set(1, n-1, 1)
	D := mat.NewDense(2, 3, nil)
	sys, _ := New(A, B, C, D, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Discretize(0.01)
	}
}

func BenchmarkTransferFunction_MIMO(b *testing.B) {
	n := 15
	A := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		A.Set(i, i, -float64(i+1)*0.3)
		if i > 0 {
			A.Set(i, i-1, 1)
		}
		if i < n-1 {
			A.Set(i, i+1, 0.1)
		}
	}
	m, p := 3, 4
	B := mat.NewDense(n, m, nil)
	for i := 0; i < m; i++ {
		B.Set(i, i, 1)
	}
	C := mat.NewDense(p, n, nil)
	for i := 0; i < p; i++ {
		C.Set(i, i%n, 1)
	}
	D := mat.NewDense(p, m, nil)
	sys, _ := New(A, B, C, D, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.TransferFunction(nil)
	}
}

func BenchmarkReduce(b *testing.B) {
	n := 20
	A := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		A.Set(i, i, -float64(i+1)*0.2)
		if i > 0 {
			A.Set(i, i-1, 1)
		}
	}
	B := mat.NewDense(n, 2, nil)
	B.Set(0, 0, 1)
	B.Set(1, 1, 1)
	C := mat.NewDense(2, n, nil)
	C.Set(0, 0, 1)
	C.Set(1, 1, 1)
	D := mat.NewDense(2, 2, nil)
	sys, _ := New(A, B, C, D, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Reduce(nil)
	}
}

func BenchmarkAbsorbDelay(b *testing.B) {
	A := mat.NewDense(5, 5, nil)
	for i := 0; i < 5; i++ {
		A.Set(i, i, 0.8-float64(i)*0.05)
		if i > 0 {
			A.Set(i, i-1, 0.1)
		}
	}
	B := mat.NewDense(5, 3, nil)
	for i := 0; i < 3; i++ {
		B.Set(i, i, 1)
	}
	C := mat.NewDense(2, 5, nil)
	C.Set(0, 0, 1)
	C.Set(1, 4, 1)
	D := mat.NewDense(2, 3, nil)
	delay := mat.NewDense(2, 3, []float64{3, 5, 3, 5, 3, 5})
	sys, _ := NewWithDelay(A, B, C, D, delay, 1.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.AbsorbDelay()
	}
}

func BenchmarkDiscretizeZOH(b *testing.B) {
	n := 20
	A := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		A.Set(i, i, -float64(i+1)*0.5)
		if i > 0 {
			A.Set(i, i-1, 1)
		}
	}
	m := 5
	B := mat.NewDense(n, m, nil)
	for i := 0; i < m; i++ {
		B.Set(i, i, 1)
	}
	C := mat.NewDense(3, n, nil)
	C.Set(0, 0, 1)
	C.Set(1, 5, 1)
	C.Set(2, n-1, 1)
	D := mat.NewDense(3, m, nil)
	sys, _ := New(A, B, C, D, 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.DiscretizeZOH(0.01)
	}
}

func BenchmarkDenseNorm(b *testing.B) {
	m := mat.NewDense(50, 50, nil)
	for i := 0; i < 50; i++ {
		for j := 0; j < 50; j++ {
			m.Set(i, j, float64(i*50+j)*0.01)
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		denseNorm(m)
	}
}

func BenchmarkSeries(b *testing.B) {
	sys1 := benchSys(10, 3, 4)
	sys2 := benchSys(8, 4, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Series(sys1, sys2)
	}
}

func BenchmarkParallel(b *testing.B) {
	sys1 := benchSys(10, 3, 4)
	sys2 := benchSys(8, 3, 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Parallel(sys1, sys2)
	}
}

func BenchmarkFeedback(b *testing.B) {
	plant := benchSys(10, 3, 3)
	ctrl := benchSys(5, 3, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Feedback(plant, ctrl, -1)
	}
}

func BenchmarkFeedbackLFT(b *testing.B) {
	plant := benchSys(10, 3, 3)
	_ = plant.SetOutputDelay([]float64{0.5, 1.0, 0.3})
	ctrl := benchSys(5, 3, 3)
	_ = ctrl.SetInputDelay([]float64{0.2, 0.4, 0.6})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Feedback(plant, ctrl, -1)
	}
}

func BenchmarkAppend(b *testing.B) {
	sys1 := benchSys(10, 3, 4)
	sys2 := benchSys(8, 2, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Append(sys1, sys2)
	}
}

func BenchmarkFreqResponse(b *testing.B) {
	sys := benchSys(10, 2, 3)
	omega := logspace(-2, 2, 100)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.FreqResponse(omega)
	}
}

func BenchmarkBode(b *testing.B) {
	sys := benchSys(10, 2, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Bode(nil, 200)
	}
}

func BenchmarkZeros_SISO(b *testing.B) {
	A := mat.NewDense(5, 5, nil)
	for i := 0; i < 5; i++ {
		A.Set(i, i, -float64(i+1)*0.5)
		if i > 0 {
			A.Set(i, i-1, 1)
		}
	}
	B := mat.NewDense(5, 1, []float64{1, 0, 0, 0, 0})
	C := mat.NewDense(1, 5, []float64{0, 0, 0, 0, 1})
	D := mat.NewDense(1, 1, []float64{0.1})
	sys, _ := New(A, B, C, D, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Zeros()
	}
}

func BenchmarkZeros_MIMO(b *testing.B) {
	sys := benchSys(10, 3, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Zeros()
	}
}

func BenchmarkZeros_NonSquare(b *testing.B) {
	n, m, p := 15, 5, 8
	A := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		A.Set(i, i, -float64(i+1)*0.3)
		if i > 0 {
			A.Set(i, i-1, 1)
		}
		if i < n-1 {
			A.Set(i, i+1, 0.1)
		}
	}
	B := mat.NewDense(n, m, nil)
	for i := 0; i < m; i++ {
		B.Set(i, i, 1)
	}
	C := mat.NewDense(p, n, nil)
	for i := 0; i < p; i++ {
		C.Set(i, i%n, 1)
	}
	D := mat.NewDense(p, m, nil)
	sys, _ := New(A, B, C, D, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Zeros()
	}
}

func BenchmarkControllabilityStaircase(b *testing.B) {
	n := 20
	A := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		A.Set(i, i, -float64(i+1)*0.2)
		if i > 0 {
			A.Set(i, i-1, 1)
		}
	}
	B := mat.NewDense(n, 3, nil)
	B.Set(0, 0, 1)
	B.Set(1, 1, 1)
	B.Set(2, 2, 1)
	C := mat.NewDense(2, n, nil)
	C.Set(0, 0, 1)
	C.Set(1, 1, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ControllabilityStaircase(A, B, C, 0)
	}
}

func BenchmarkSimulateNoDelay(b *testing.B) {
	sys := benchSys(20, 3, 4)
	sys.Dt = 0.01
	steps := 500
	u := mat.NewDense(3, steps, nil)
	for j := 0; j < 3; j++ {
		for k := 0; k < steps; k++ {
			u.Set(j, k, math.Sin(float64(k)*0.1+float64(j)))
		}
	}
	x0 := mat.NewVecDense(20, nil)
	for i := 0; i < 20; i++ {
		x0.SetVec(i, float64(i)*0.05)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Simulate(u, x0, nil)
	}
}

func BenchmarkThiranDelay(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ThiranDelay(0.35, 3, 0.1)
	}
}

func BenchmarkPadeDelay(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		PadeDelay(0.5, 5)
	}
}

func BenchmarkDecomposeIODelay(b *testing.B) {
	delay := mat.NewDense(4, 3, []float64{
		3, 5, 2,
		4, 6, 3,
		3, 5, 2,
		5, 7, 4,
	})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DecomposeIODelay(delay)
	}
}

func BenchmarkPullDelaysToLFT(b *testing.B) {
	sys := benchSys(5, 3, 2)
	sys.InputDelay = []float64{0.3, 0.5, 0.1}
	sys.OutputDelay = []float64{0.2, 0.4}
	sys.Delay = mat.NewDense(2, 3, []float64{1, 2, 1, 2, 3, 2})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.PullDelaysToLFT()
	}
}

func BenchmarkGetDelayModel(b *testing.B) {
	sys := benchSys(5, 2, 2)
	sys.InputDelay = []float64{0.3, 0.5}
	sys.OutputDelay = []float64{0.2, 0.4}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.GetDelayModel()
	}
}

func BenchmarkSetDelayModel(b *testing.B) {
	sys := benchSys(5, 2, 2)
	sys.InputDelay = []float64{0.3, 0.5}
	H, tau := sys.GetDelayModel()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SetDelayModel(H, tau)
	}
}

func BenchmarkAbsorbInternalDelay(b *testing.B) {
	A := mat.NewDense(5, 5, nil)
	for i := 0; i < 5; i++ {
		A.Set(i, i, 0.8-float64(i)*0.05)
		if i > 0 {
			A.Set(i, i-1, 0.1)
		}
	}
	B := mat.NewDense(5, 2, nil)
	B.Set(0, 0, 1)
	B.Set(1, 1, 1)
	C := mat.NewDense(2, 5, nil)
	C.Set(0, 0, 1)
	C.Set(1, 4, 1)
	D := mat.NewDense(2, 2, nil)
	sys, _ := New(A, B, C, D, 1.0)
	B2 := mat.NewDense(5, 2, []float64{0.5, 0, 0, 0.3, 0.1, 0, 0, 0.2, 0, 0.1})
	C2 := mat.NewDense(2, 5, []float64{0.2, 0.4, 0, 0, 0, 0, 0, 0.3, 0.1, 0})
	D12 := mat.NewDense(2, 2, nil)
	D21 := mat.NewDense(2, 2, nil)
	D22 := mat.NewDense(2, 2, nil)
	sys.SetInternalDelay([]float64{3, 5}, B2, C2, D12, D21, D22)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.AbsorbDelay(AbsorbInternal)
	}
}

func BenchmarkDiscretizeWithOpts_Thiran(b *testing.B) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.InputDelay = []float64{0.35}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.DiscretizeWithOpts(0.1, C2DOptions{Method: "zoh", ThiranOrder: 3})
	}
}

func BenchmarkDiscretizeWithOpts_IODelayThiran(b *testing.B) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys.Delay = mat.NewDense(1, 1, []float64{0.35})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.DiscretizeWithOpts(0.1, C2DOptions{Method: "zoh", ThiranOrder: 3})
	}
}

func BenchmarkSafeFeedback(b *testing.B) {
	plant := benchSys(10, 3, 3)
	plant.Dt = 1.0
	ctrl := benchSys(5, 3, 3)
	ctrl.Dt = 1.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SafeFeedback(plant, ctrl, -1)
	}
}

func BenchmarkSimulateInternalDelay(b *testing.B) {
	A := mat.NewDense(3, 3, []float64{0.9, 0.1, 0, 0, 0.8, 0.1, 0, 0, 0.7})
	B := mat.NewDense(3, 1, []float64{1, 0, 0})
	C := mat.NewDense(1, 3, []float64{1, 0, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 1.0)
	B2 := mat.NewDense(3, 1, []float64{0.5, 0.3, 0.1})
	C2 := mat.NewDense(1, 3, []float64{0.2, 0.4, 0})
	D12 := mat.NewDense(1, 1, []float64{0})
	D21 := mat.NewDense(1, 1, []float64{0})
	D22 := mat.NewDense(1, 1, []float64{0})
	sys.SetInternalDelay([]float64{3}, B2, C2, D12, D21, D22)

	steps := 200
	u := mat.NewDense(1, steps, nil)
	for k := 0; k < steps; k++ {
		u.Set(0, k, math.Sin(float64(k)*0.1))
	}
	x0 := mat.NewVecDense(3, []float64{1, 0, 0})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Simulate(u, x0, nil)
	}
}

func BenchmarkFreqResponseWithDelay(b *testing.B) {
	sys := benchSys(5, 2, 2)
	sys.InputDelay = []float64{0.3, 0.5}
	sys.OutputDelay = []float64{0.2, 0.4}
	omega := logspace(-2, 2, 100)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.FreqResponse(omega)
	}
}

func BenchmarkPadeSystem(b *testing.B) {
	sys := benchSys(5, 2, 2)
	sys.InputDelay = []float64{0.3, 0.5}
	sys.OutputDelay = []float64{0.2, 0.4}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.Pade(3)
	}
}

func BenchmarkZeroDelayApprox(b *testing.B) {
	sys := benchSys(5, 2, 2)
	sys.Dt = 1.0
	B2 := mat.NewDense(5, 2, []float64{0.5, 0, 0, 0.3, 0.1, 0, 0, 0.2, 0, 0.1})
	C2 := mat.NewDense(2, 5, []float64{0.2, 0.4, 0, 0, 0, 0, 0, 0.3, 0.1, 0})
	D12 := mat.NewDense(2, 2, nil)
	D21 := mat.NewDense(2, 2, nil)
	D22 := mat.NewDense(2, 2, nil)
	sys.SetInternalDelay([]float64{3, 5}, B2, C2, D12, D21, D22)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.ZeroDelayApprox()
	}
}

func BenchmarkIsStrictlyUpperTriangular(b *testing.B) {
	n := 20
	m := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			m.Set(i, j, float64(i*n+j)*0.01)
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		isStrictlyUpperTriangular(m)
	}
}

func BenchmarkFreqResponseLFT(b *testing.B) {
	sys := benchSys(5, 2, 2)
	sys.Dt = 1.0
	B2 := mat.NewDense(5, 2, []float64{0.5, 0, 0, 0.3, 0.1, 0, 0, 0.2, 0, 0.1})
	C2 := mat.NewDense(2, 5, []float64{0.2, 0.4, 0, 0, 0, 0, 0, 0.3, 0.1, 0})
	D12 := mat.NewDense(2, 2, nil)
	D21 := mat.NewDense(2, 2, nil)
	D22 := mat.NewDense(2, 2, nil)
	sys.SetInternalDelay([]float64{3, 5}, B2, C2, D12, D21, D22)
	omega := logspace(-2, 2, 100)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sys.FreqResponse(omega)
	}
}

func benchSys(n, m, p int) *System {
	A := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		A.Set(i, i, -float64(i+1)*0.3)
		if i > 0 {
			A.Set(i, i-1, 1)
		}
		if i < n-1 {
			A.Set(i, i+1, 0.1)
		}
	}
	B := mat.NewDense(n, m, nil)
	for i := 0; i < min(n, m); i++ {
		B.Set(i, i, 1)
	}
	C := mat.NewDense(p, n, nil)
	for i := 0; i < min(p, n); i++ {
		C.Set(i, i, 1)
	}
	D := mat.NewDense(p, m, nil)
	sys, _ := New(A, B, C, D, 0)
	return sys
}
