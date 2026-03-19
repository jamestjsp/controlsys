package controlsys

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func (sys *System) Pade(order int) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("Pade: continuous only, use AbsorbDelay for discrete: %w", ErrWrongDomain)
	}
	if !sys.HasDelay() {
		return sys.Copy(), nil
	}

	lft, err := sys.PullDelaysToLFT()
	if err != nil {
		return nil, fmt.Errorf("Pade: %w", err)
	}

	N := len(lft.InternalDelay)
	if N == 0 {
		return sys.Copy(), nil
	}

	n, m, p := lft.Dims()

	var delayBank *System
	for j := 0; j < N; j++ {
		pd, err := PadeDelay(lft.InternalDelay[j], order)
		if err != nil {
			return nil, fmt.Errorf("Pade: delay %d (tau=%v): %w", j, lft.InternalDelay[j], err)
		}
		if delayBank == nil {
			delayBank = pd
		} else {
			delayBank, err = Append(delayBank, pd)
			if err != nil {
				return nil, err
			}
		}
	}

	nd, _, _ := delayBank.Dims()
	nTotal := n + nd

	B2 := lft.B2
	C2 := lft.C2
	D12 := lft.D12
	D21 := lft.D21
	D22 := lft.D22

	Dd := delayBank.D
	D22Dd := mat.NewDense(N, N, nil)
	D22Dd.Mul(D22, Dd)
	E := mat.NewDense(N, N, nil)
	eRaw := E.RawMatrix()
	for i := 0; i < N; i++ {
		eRaw.Data[i*eRaw.Stride+i] = 1
	}
	E.Sub(E, D22Dd)
	var lu mat.LU
	lu.Factorize(E)
	D22Dd.Zero()
	idRaw := D22Dd.RawMatrix()
	for i := 0; i < N; i++ {
		idRaw.Data[i*idRaw.Stride+i] = 1
	}
	Einv := mat.NewDense(N, N, nil)
	if err := lu.SolveTo(Einv, false, D22Dd); err != nil {
		return nil, fmt.Errorf("Pade: (I - D22*Dd) singular: %w", ErrSingularTransform)
	}

	DdE := mat.NewDense(N, N, nil)
	DdE.Mul(Dd, Einv)

	var DdEC2 *mat.Dense
	if n > 0 {
		DdEC2 = mat.NewDense(N, n, nil)
		DdEC2.Mul(DdE, C2)
	}

	DdED21 := mat.NewDense(N, m, nil)
	DdED21.Mul(DdE, D21)

	var D22Cd *mat.Dense
	DdED22Cd := mat.NewDense(N, nd, nil)
	if nd > 0 {
		D22Cd = mat.NewDense(N, nd, nil)
		D22Cd.Mul(D22, delayBank.C)
		DdED22Cd.Mul(DdE, D22Cd)
	}

	DdED22CdPlusCd := mat.NewDense(N, nd, nil)
	if nd > 0 {
		DdED22CdPlusCd.Add(DdED22Cd, delayBank.C)
	}

	BdE := mat.NewDense(nd, N, nil)
	if nd > 0 {
		BdE.Mul(delayBank.B, Einv)
	}

	Acl := mat.NewDense(nTotal, nTotal, nil)
	Bcl := mat.NewDense(nTotal, m, nil)
	Ccl := mat.NewDense(p, nTotal, nil)
	Dcl := mat.NewDense(p, m, nil)

	if n > 0 {
		setBlock(Acl, 0, 0, lft.A)
		tmp := mat.NewDense(n, n, nil)
		tmp.Mul(B2, DdEC2)
		addBlock(Acl, 0, 0, tmp)

		if nd > 0 {
			tmp2 := mat.NewDense(n, nd, nil)
			tmp2.Mul(B2, DdED22CdPlusCd)
			setBlock(Acl, 0, n, tmp2)
		}

		setBlock(Bcl, 0, 0, lft.B)
		tmp3 := mat.NewDense(n, m, nil)
		tmp3.Mul(B2, DdED21)
		addBlock(Bcl, 0, 0, tmp3)
	}

	if nd > 0 {
		if n > 0 {
			tmp := mat.NewDense(nd, n, nil)
			tmp.Mul(BdE, C2)
			setBlock(Acl, n, 0, tmp)
		}

		setBlock(Acl, n, n, delayBank.A)
		tmp2 := mat.NewDense(nd, nd, nil)
		tmp2.Mul(BdE, D22Cd)
		addBlock(Acl, n, n, tmp2)

		tmp3 := mat.NewDense(nd, m, nil)
		tmp3.Mul(BdE, D21)
		setBlock(Bcl, n, 0, tmp3)
	}

	if n > 0 {
		setBlock(Ccl, 0, 0, lft.C)
		tmp := mat.NewDense(p, n, nil)
		tmp.Mul(D12, DdEC2)
		addBlock(Ccl, 0, 0, tmp)
	}
	if nd > 0 {
		tmp := mat.NewDense(p, nd, nil)
		tmp.Mul(D12, DdED22CdPlusCd)
		setBlock(Ccl, 0, n, tmp)
	}

	Dcl.Mul(D12, DdED21)
	Dcl.Add(Dcl, lft.D)

	return newNoCopy(Acl, Bcl, Ccl, Dcl, 0)
}

func PadeDelay(tau float64, order int) (*System, error) {
	if tau < 0 {
		return nil, ErrNegativeDelay
	}
	if order < 1 || order > 10 {
		return nil, fmt.Errorf("PadeDelay: order must be 1-10: %w", ErrDimensionMismatch)
	}
	if tau == 0 {
		return NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	}

	n := order

	// q_k = (2n-k)! * n! / ((2n)! * k! * (n-k)!)
	// Recurrence: q_k = q_{k-1} * (n-k+1) / (k * (2n-k+1))
	q := make([]float64, n+1)
	q[0] = 1.0
	for k := 1; k <= n; k++ {
		q[k] = q[k-1] * float64(n-k+1) / float64(k*(2*n-k+1))
	}

	// den(s) = sum_{k=0}^n q_k * (tau*s)^k, descending order
	// num(s) = sum_{k=0}^n (-1)^k * q_k * (tau*s)^k, descending order
	den := make([]float64, n+1)
	num := make([]float64, n+1)
	for k := 0; k <= n; k++ {
		tauPow := math.Pow(tau, float64(k))
		den[n-k] = q[k] * tauPow
		sign := 1.0
		if k%2 != 0 {
			sign = -1.0
		}
		num[n-k] = sign * q[k] * tauPow
	}

	tf := &TransferFunc{
		Num: [][][]float64{{num}},
		Den: [][]float64{den},
		Dt:  0,
	}

	result, err := tf.StateSpace(nil)
	if err != nil {
		return nil, fmt.Errorf("PadeDelay: %w", err)
	}
	return result.Sys, nil
}
