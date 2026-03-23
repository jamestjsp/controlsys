package controlsys

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func feedbackWithLFT(plant, controller *System, sign float64) (*System, error) {
	savedPlantInput := copySliceOrNil(plant.InputDelay)

	plantStripped := plant.Copy()
	plantStripped.InputDelay = nil

	plantLFT, err := plantStripped.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}
	ctrlLFT, err := controller.PullDelaysToLFT()
	if err != nil {
		return nil, err
	}

	var plantTau, ctrlTau []float64
	if plantLFT.LFT != nil {
		plantTau = plantLFT.LFT.Tau
	}
	if ctrlLFT.LFT != nil {
		ctrlTau = ctrlLFT.LFT.Tau
	}

	pH, _ := plantLFT.GetDelayModel()
	cH, _ := ctrlLFT.GetDelayModel()

	Np := len(plantTau)
	Nc := len(ctrlTau)
	Nd := Np + Nc

	_, m1, p1 := plant.Dims()
	np, _, _ := pH.Dims()
	nc, _, _ := cH.Dims()

	isign := -sign
	nTotal := np + nc
	mTotal := m1 + Nd
	pTotal := p1 + Nd

	A := mat.NewDense(max(nTotal, 1), max(nTotal, 1), nil)
	B := mat.NewDense(max(nTotal, 1), max(mTotal, 1), nil)
	C := mat.NewDense(max(pTotal, 1), max(nTotal, 1), nil)
	D := mat.NewDense(max(pTotal, 1), max(mTotal, 1), nil)

	D1ee := extractBlock(pH.D, 0, 0, p1, m1)
	D2ee := extractBlock(cH.D, 0, 0, m1, p1)

	M := mat.NewDense(p1, p1, nil)
	M.Mul(D1ee, D2ee)
	M.Scale(isign, M)
	mRaw := M.RawMatrix()
	for i := 0; i < p1; i++ {
		mRaw.Data[i*mRaw.Stride+i] += 1
	}

	var lu mat.LU
	lu.Factorize(M)
	if lu.Det() == 0 {
		return nil, fmt.Errorf("feedback: (I + sign*D1*D2) singular: %w", ErrSingularTransform)
	}

	E21 := mat.NewDense(p1, p1, nil)
	if err := lu.SolveTo(E21, false, eyeDense(p1)); err != nil {
		return nil, fmt.Errorf("feedback: LU solve failed: %w", ErrSingularTransform)
	}

	E12 := eyeDense(m1)
	{
		t := mulDense(mulDense(D2ee, E21), D1ee)
		t.Scale(isign, t)
		E12.Sub(E12, t)
	}

	B1e := extractBlock(pH.B, 0, 0, np, m1)
	C1e := extractBlock(pH.C, 0, 0, p1, np)
	B2e := extractBlock(cH.B, 0, 0, nc, p1)
	C2e := extractBlock(cH.C, 0, 0, m1, nc)

	E21D1ee := mulDense(E21, D1ee)

	var B1eE12, B1eE12D2ee *mat.Dense
	if np > 0 {
		B1eE12 = mulDense(B1e, E12)
		B1eE12D2ee = mulDense(B1eE12, D2ee)
	}

	var B2eE21, B2eE21D1ee *mat.Dense
	if nc > 0 {
		B2eE21 = mulDense(B2e, E21)
		B2eE21D1ee = mulDense(B2eE21, D1ee)
	}

	var E21C1e *mat.Dense
	if np > 0 {
		E21C1e = mulDense(E21, C1e)
	}

	var D1de, D1ed, E21D1ed *mat.Dense
	if Np > 0 {
		D1de = extractBlock(pH.D, p1, 0, Np, m1)
		D1ed = extractBlock(pH.D, 0, m1, p1, Np)
		E21D1ed = mulDense(E21, D1ed)
	}

	var D2de, D2ed *mat.Dense
	if Nc > 0 {
		D2de = extractBlock(cH.D, m1, 0, Nc, p1)
		D2ed = extractBlock(cH.D, 0, p1, m1, Nc)
	}

	var D1deD2ee *mat.Dense
	if Np > 0 {
		D1deD2ee = mulDense(D1de, D2ee)
	}

	var D2deE21, D2deE21D1ee *mat.Dense
	if Nc > 0 {
		D2deE21 = mulDense(D2de, E21)
		D2deE21D1ee = mulDense(D2deE21, D1ee)
	}

	var E12D2ed *mat.Dense
	if Nc > 0 {
		E12D2ed = mulDense(E12, D2ed)
	}

	if np > 0 {
		setBlock(A, 0, 0, pH.A)
		t := mulDense(B1eE12D2ee, C1e)
		t.Scale(isign, t)
		subBlock(A, 0, 0, t)
		if nc > 0 {
			t2 := mulDense(B1eE12, C2e)
			t2.Scale(-isign, t2)
			setBlock(A, 0, np, t2)
		}
	}
	if nc > 0 {
		if np > 0 {
			setBlock(A, np, 0, mulDense(B2eE21, C1e))
		}
		setBlock(A, np, np, cH.A)
		t := mulDense(B2eE21D1ee, C2e)
		t.Scale(isign, t)
		subBlock(A, np, np, t)
	}

	if np > 0 {
		setBlock(B, 0, 0, B1eE12)
	}
	if nc > 0 {
		setBlock(B, np, 0, B2eE21D1ee)
	}

	if Np > 0 {
		B1d := extractBlock(pH.B, 0, m1, np, Np)
		if np > 0 {
			t := mulDense(B1eE12D2ee, E21D1ed)
			t.Scale(-isign, t)
			t.Add(B1d, t)
			setBlock(B, 0, m1, t)
		}
		if nc > 0 {
			setBlock(B, np, m1, mulDense(B2eE21, D1ed))
		}
	}

	if Nc > 0 {
		B2d := extractBlock(cH.B, 0, p1, nc, Nc)
		if np > 0 {
			t := mulDense(B1eE12, D2ed)
			t.Scale(-isign, t)
			setBlock(B, 0, m1+Np, t)
		}
		if nc > 0 {
			t := mulDense(mulDense(B2eE21D1ee, E12), D2ed)
			t.Scale(isign, t)
			t.Add(B2d, t)
			setBlock(B, np, m1+Np, t)
		}
	}

	if np > 0 {
		setBlock(C, 0, 0, E21C1e)
	}
	if nc > 0 {
		t := mulDense(E21D1ee, C2e)
		t.Scale(-isign, t)
		setBlock(C, 0, np, t)
	}

	if Np > 0 {
		C1d := extractBlock(pH.C, p1, 0, Np, np)
		if np > 0 {
			t := mulDense(D1deD2ee, E21C1e)
			t.Scale(isign, t)
			r := mat.NewDense(Np, np, nil)
			r.Sub(C1d, t)
			setBlock(C, p1, 0, r)
		}
		if nc > 0 {
			t := mulDense(mulDense(D1deD2ee, E21D1ee), C2e)
			t.Scale(-isign, t)
			setBlock(C, p1, np, t)
		}
	}

	if Nc > 0 {
		C2d := extractBlock(cH.C, m1, 0, Nc, nc)
		if np > 0 {
			setBlock(C, p1+Np, 0, mulDense(D2deE21, C1e))
		}
		if nc > 0 {
			t := mulDense(D2deE21D1ee, C2e)
			t.Scale(-isign, t)
			t.Add(C2d, t)
			setBlock(C, p1+Np, np, t)
		}
	}

	setBlock(D, 0, 0, E21D1ee)

	if Np > 0 {
		setBlock(D, 0, m1, E21D1ed)
	}
	if Nc > 0 {
		t := mulDense(E21D1ee, E12D2ed)
		t.Scale(-isign, t)
		setBlock(D, 0, m1+Np, t)
	}

	if Np > 0 {
		D1dd := extractBlock(pH.D, p1, m1, Np, Np)

		setBlock(D, p1, 0, mulDense(D1de, E12))

		t := mulDense(D1deD2ee, E21D1ed)
		t.Scale(isign, t)
		r := mat.NewDense(Np, Np, nil)
		r.Sub(D1dd, t)
		setBlock(D, p1, m1, r)

		if Nc > 0 {
			t := mulDense(D1de, E12D2ed)
			t.Scale(-isign, t)
			setBlock(D, p1, m1+Np, t)
		}
	}

	if Nc > 0 {
		D2dd := extractBlock(cH.D, m1, p1, Nc, Nc)

		setBlock(D, p1+Np, 0, D2deE21D1ee)

		if Np > 0 {
			setBlock(D, p1+Np, m1, mulDense(D2deE21, D1ed))
		}

		t := mulDense(D2deE21D1ee, E12D2ed)
		t.Scale(-isign, t)
		t.Add(D2dd, t)
		setBlock(D, p1+Np, m1+Np, t)
	}

	if nTotal == 0 {
		A = &mat.Dense{}
	} else {
		A = resizeDense(A, nTotal, nTotal)
	}
	B = resizeDense(B, nTotal, mTotal)
	C = resizeDense(C, pTotal, nTotal)
	D = resizeDense(D, pTotal, mTotal)

	H := &System{A: A, B: B, C: C, D: D, Dt: plant.Dt}

	taus := make([]float64, Nd)
	copy(taus, plantTau)
	copy(taus[Np:], ctrlTau)

	result, err := SetDelayModel(H, taus)
	if err != nil {
		return nil, err
	}
	result.InputDelay = savedPlantInput
	result.InputName = copyStringSlice(plant.InputName)
	result.OutputName = copyStringSlice(plant.OutputName)
	return result, nil
}

func eyeDense(n int) *mat.Dense {
	data := make([]float64, n*n)
	for i := 0; i < n; i++ {
		data[i*(n+1)] = 1
	}
	return mat.NewDense(n, n, data)
}
