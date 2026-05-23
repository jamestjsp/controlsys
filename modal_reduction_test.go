package controlsys

import "testing"

func TestModalTruncateReducesOrderAndPreservesMetadata(t *testing.T) {
	sys := benchSysNonSym(4, 1, 1)
	sys.InputName = []string{"u"}
	sys.OutputName = []string{"y"}
	sys.StateName = []string{"x1", "x2", "x3", "x4"}

	result, err := ModalTruncate(sys, &ModalTruncateOptions{Order: 2})
	if err != nil {
		t.Fatalf("ModalTruncate: %v", err)
	}
	if result.Method != "modal-truncate" || result.Order != 2 {
		t.Fatalf("metadata = %#v", result)
	}
	if n, _, _ := result.Sys.Dims(); n != 2 {
		t.Fatalf("reduced order = %d, want 2", n)
	}
	if !sameStrings(result.Sys.InputName, []string{"u"}) || !sameStrings(result.Sys.OutputName, []string{"y"}) {
		t.Fatalf("names = %v/%v", result.Sys.InputName, result.Sys.OutputName)
	}
}

func TestModalTruncateAutoOrderAndInvalidOptions(t *testing.T) {
	sys := benchSysNonSym(4, 1, 1)
	result, err := ModalTruncate(sys, &ModalTruncateOptions{MaxRealPart: -1.0})
	if err != nil {
		t.Fatalf("ModalTruncate auto: %v", err)
	}
	if result.Order == 0 || result.Order >= 4 {
		t.Fatalf("auto order = %d, want partial reduction", result.Order)
	}
	if _, err := ModalTruncate(sys, &ModalTruncateOptions{Order: 5}); err != ErrInvalidOrder {
		t.Fatalf("invalid order err = %v, want ErrInvalidOrder", err)
	}
}
