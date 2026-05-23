package controlsys

import (
	"errors"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestPhysicalAssemblyGroundedDescriptorComponent(t *testing.T) {
	component := physicalTestComponent(t, "mass")
	asm, err := AssemblePhysical("grounded", []PhysicalComponent{component}, []PhysicalConnection{
		{FromComponent: "mass", FromPort: "mount", Grounded: true},
	})
	if err != nil {
		t.Fatalf("AssemblePhysical: %v", err)
	}
	if !asm.IsDescriptor() {
		t.Fatal("assembled descriptor component should remain descriptor")
	}
	if !matEqual(asm.A, component.System.A, 1e-12) || !matEqual(asm.E, component.System.E, 1e-12) {
		t.Fatalf("assembled matrices do not match component")
	}
	if !sameStrings(asm.StateName, []string{"mass.x1", "mass.x2"}) {
		t.Fatalf("state names = %v", asm.StateName)
	}
}

func TestPhysicalAssemblyTwoComponentCoupling(t *testing.T) {
	left := physicalTestComponent(t, "left")
	right := physicalTestComponent(t, "right")
	asm, err := AssemblePhysical("pair", []PhysicalComponent{left, right}, []PhysicalConnection{
		{FromComponent: "left", FromPort: "mount", ToComponent: "right", ToPort: "mount"},
	})
	if err != nil {
		t.Fatalf("AssemblePhysical: %v", err)
	}
	if n, m, p := asm.Dims(); n != 4 || m != 2 || p != 2 {
		t.Fatalf("dims = (%d,%d,%d), want (4,2,2)", n, m, p)
	}
	if asm.A.At(0, 1) != left.System.A.At(0, 1) || asm.A.At(2, 3) != right.System.A.At(0, 1) {
		t.Fatalf("block diagonal A not preserved: %v", mat.Formatted(asm.A))
	}
	if !sameStrings(asm.InputName, []string{"left.force", "right.force"}) {
		t.Fatalf("input names = %v", asm.InputName)
	}
}

func TestPhysicalAssemblyRejectsIncompatiblePorts(t *testing.T) {
	left := physicalTestComponent(t, "left")
	right := physicalTestComponent(t, "right")
	right.Ports[0].Kind = PhysicalPortEffort
	_, err := AssemblePhysical("bad", []PhysicalComponent{left, right}, []PhysicalConnection{
		{FromComponent: "left", FromPort: "mount", ToComponent: "right", ToPort: "mount"},
	})
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("incompatible port err = %v, want ErrDimensionMismatch", err)
	}
}

func physicalTestComponent(t *testing.T, name string) PhysicalComponent {
	t.Helper()
	sys, err := NewDescriptor(
		mat.NewDense(2, 2, []float64{-1, 0.5, -2, -3}),
		mat.NewDense(2, 1, []float64{1, -1}),
		mat.NewDense(1, 2, []float64{2, -0.5}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(2, 2, []float64{2, 0, 0.1, 3}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputName = []string{"force"}
	sys.OutputName = []string{"position"}
	sys.StateName = []string{"x1", "x2"}
	return NewPhysicalComponent(name, sys, []PhysicalPort{{Name: "mount", Kind: PhysicalPortDisplacement, Dimension: 1}})
}
