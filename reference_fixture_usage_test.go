package controlsys

import "testing"

func TestReferenceFixtureBuildsPythonControlStateSpaceModel(t *testing.T) {
	ref := pythonControlReference(t, "state-space fixture smoke test", RefTolTight)
	sys := ref.StateSpace(
		2, 1, 1,
		[]float64{0, 1, -2, -3},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{0},
		0,
	)

	n, m, p := sys.Dims()
	if n != 2 || m != 1 || p != 1 {
		t.Fatalf("dims = (%d,%d,%d), want (2,1,1)", n, m, p)
	}
	ref.AssertDense("A", sys.A, ref.Dense(2, 2, []float64{0, 1, -2, -3}))
}
