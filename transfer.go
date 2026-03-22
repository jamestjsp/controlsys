package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/blas"
	gonumLapack "gonum.org/v1/gonum/lapack/gonum"
	"gonum.org/v1/gonum/mat"
)

var impl gonumLapack.Implementation

type TransferFunc struct {
	Num   [][][]float64
	Den   [][]float64
	Delay [][]float64 // [p][m]; nil = no delay
	Dt    float64
}

func (tf *TransferFunc) Dims() (p, m int) {
	p = len(tf.Den)
	if p > 0 && len(tf.Num) > 0 && len(tf.Num[0]) > 0 {
		m = len(tf.Num[0])
	}
	return
}

func (tf *TransferFunc) Eval(s complex128) [][]complex128 {
	p, m := tf.Dims()
	result := make([][]complex128, p)
	for i := 0; i < p; i++ {
		result[i] = make([]complex128, m)
		dv := Poly(tf.Den[i]).Eval(s)
		for j := 0; j < m; j++ {
			nv := Poly(tf.Num[i][j]).Eval(s)
			h := nv / dv
			if tf.Delay != nil && tf.Delay[i][j] != 0 {
				tau := tf.Delay[i][j]
				if tf.Dt == 0 {
					h *= cmplx.Exp(-s * complex(tau, 0))
				} else {
					d := int(math.Round(tau))
					for k := 0; k < d; k++ {
						h /= s
					}
				}
			}
			result[i][j] = h
		}
	}
	return result
}

func (tf *TransferFunc) evalInto(s complex128, dst []complex128) {
	p, m := tf.Dims()
	for i := 0; i < p; i++ {
		dv := Poly(tf.Den[i]).Eval(s)
		for j := 0; j < m; j++ {
			h := Poly(tf.Num[i][j]).Eval(s) / dv
			if tf.Delay != nil && tf.Delay[i][j] != 0 {
				tau := tf.Delay[i][j]
				if tf.Dt == 0 {
					h *= cmplx.Exp(-s * complex(tau, 0))
				} else {
					d := int(math.Round(tau))
					for k := 0; k < d; k++ {
						h /= s
					}
				}
			}
			dst[i*m+j] = h
		}
	}
}

func (tf *TransferFunc) EvalMulti(freqs []complex128) [][][]complex128 {
	result := make([][][]complex128, len(freqs))
	for k, s := range freqs {
		result[k] = tf.Eval(s)
	}
	return result
}

type TransferFuncOpts struct {
	ControllabilityTol float64
	ObservabilityTol   float64
}

type TransferFuncResult struct {
	TF           *TransferFunc
	MinimalOrder int
	RowDegrees   []int
}

func (sys *System) TransferFunction(opts *TransferFuncOpts) (*TransferFuncResult, error) {
	n, m, p := sys.Dims()

	if opts == nil {
		opts = &TransferFuncOpts{}
	}

	tf := &TransferFunc{
		Num: make([][][]float64, p),
		Den: make([][]float64, p),
		Dt:  sys.Dt,
	}
	rowDeg := make([]int, p)

	if n == 0 || p == 0 || m == 0 {
		for i := 0; i < p; i++ {
			tf.Den[i] = []float64{1}
			tf.Num[i] = make([][]float64, m)
			for j := 0; j < m; j++ {
				if p > 0 && m > 0 {
					tf.Num[i][j] = []float64{sys.D.At(i, j)}
				} else {
					tf.Num[i][j] = []float64{0}
				}
			}
		}
		return &TransferFuncResult{TF: tf, MinimalOrder: 0, RowDegrees: rowDeg}, nil
	}

	stair := ControllabilityStaircase(sys.A, sys.B, sys.C, opts.ControllabilityTol)
	ncont := stair.NCont

	if ncont == 0 {
		for i := 0; i < p; i++ {
			tf.Den[i] = []float64{1}
			tf.Num[i] = make([][]float64, m)
			for j := 0; j < m; j++ {
				tf.Num[i][j] = []float64{sys.D.At(i, j)}
			}
		}
		return &TransferFuncResult{TF: tf, MinimalOrder: 0, RowDegrees: rowDeg}, nil
	}

	Ac := extractSubmatrix(stair.A, 0, ncont, 0, ncont)
	Bc := extractSubmatrix(stair.B, 0, ncont, 0, m)
	Cc := extractSubmatrix(stair.C, 0, p, 0, ncont)

	var ws *tfWorkspace
	if ncont > 0 {
		ws = newTFWorkspace(ncont, m)
	}

	totalOrder := 0
	for i := 0; i < p; i++ {
		nobs, den, numCoeffs := ssToTFRow(Ac, Bc, Cc, i, ncont, m, opts.ObservabilityTol, ws)
		rowDeg[i] = nobs
		totalOrder += nobs

		tf.Den[i] = den
		tf.Num[i] = make([][]float64, m)
		for j := 0; j < m; j++ {
			num := numCoeffs[j]
			dij := sys.D.At(i, j)
			if dij != 0 {
				numWithD := Poly(num).Add(Poly(den).Scale(dij))
				tf.Num[i][j] = []float64(numWithD)
			} else {
				tf.Num[i][j] = num
			}
		}
	}

	if sys.Delay != nil {
		tf.Delay = denseToSlice2D(sys.Delay)
	}

	return &TransferFuncResult{TF: tf, MinimalOrder: totalOrder, RowDegrees: rowDeg}, nil
}

type tfWorkspace struct {
	aData    []float64
	bsimo    []float64
	cData    []float64
	workDlrf []float64
	workC    []float64
	tauHrd   []float64
	workHrd  []float64
	workOrm  []float64
}

func newTFWorkspace(n, m int) *tfWorkspace {
	ws := &tfWorkspace{
		aData:    make([]float64, n*n),
		bsimo:    make([]float64, n),
		cData:    make([]float64, m*n),
		workDlrf: make([]float64, max(n, 1)),
		workC:    make([]float64, max(m, 1)),
	}
	if n > 1 {
		ws.tauHrd = make([]float64, n-1)
		workQuery := make([]float64, 1)
		impl.Dgehrd(n, 0, n-1, ws.aData, n, ws.tauHrd, workQuery, -1)
		lwork := int(workQuery[0])
		impl.Dormhr(blas.Right, blas.NoTrans, m, n, 0, n-1, ws.aData, n, ws.tauHrd, ws.cData, n, workQuery, -1)
		if l2 := int(workQuery[0]); l2 > lwork {
			lwork = l2
		}
		ws.workHrd = make([]float64, lwork)
		ws.workOrm = ws.workHrd
	}
	return ws
}

func formDualSIMO(Ac, Bc, Cc *mat.Dense, row, n, m int, ws *tfWorkspace) {
	aData := ws.aData
	acRaw := Ac.RawMatrix()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			aData[i*n+j] = acRaw.Data[j*acRaw.Stride+i]
		}
	}
	ccRaw := Cc.RawMatrix()
	copy(ws.bsimo, ccRaw.Data[row*ccRaw.Stride:row*ccRaw.Stride+n])
	cData := ws.cData
	bcRaw := Bc.RawMatrix()
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			cData[i*n+j] = bcRaw.Data[j*bcRaw.Stride+i]
		}
	}
}

func observabilityReduction(n, m int, ws *tfWorkspace, obsTol float64) (nobs int, b1, scaleB float64) {
	aData := ws.aData
	bsimo := ws.bsimo
	cData := ws.cData

	smlnum := math.SmallestNonzeroFloat64 / eps()
	bignum := 1.0 / smlnum

	maxA := 0.0
	for _, v := range aData[:n*n] {
		if a := math.Abs(v); a > maxA {
			maxA = a
		}
	}
	maxB := 0.0
	for _, v := range bsimo[:n] {
		if a := math.Abs(v); a > maxB {
			maxB = a
		}
	}

	if maxB == 0 {
		return 0, 0, 1
	}

	scaleA := 1.0
	if maxA > 0 && maxA < smlnum {
		scaleA = smlnum / maxA
	} else if maxA > bignum {
		scaleA = bignum / maxA
	}
	if scaleA != 1.0 {
		for i := range aData[:n*n] {
			aData[i] *= scaleA
		}
	}

	scaleB = 1.0
	if maxB < smlnum {
		scaleB = smlnum / maxB
	} else if maxB > bignum {
		scaleB = bignum / maxB
	}
	if scaleB != 1.0 {
		for i := range bsimo[:n] {
			bsimo[i] *= scaleB
		}
	}

	frobA := 0.0
	for _, v := range aData[:n*n] {
		frobA += v * v
	}
	frobA = math.Sqrt(frobA)

	norm1B := 0.0
	for _, v := range bsimo[:n] {
		norm1B += math.Abs(v)
	}

	tol := obsTol
	if tol == 0 {
		tol = float64(n) * eps() * math.Max(frobA, norm1B)
	}

	if norm1B <= tol {
		return 0, 0, scaleB
	}

	beta, tau := impl.Dlarfg(n, bsimo[0], bsimo[1:], 1)
	bsimo[0] = 1.0 // v[0]=1 for reflector

	impl.Dlarf(blas.Right, n, n, bsimo, 1, tau, aData, n, ws.workDlrf)
	impl.Dlarf(blas.Left, n, n, bsimo, 1, tau, aData, n, ws.workDlrf)
	impl.Dlarf(blas.Right, m, n, bsimo, 1, tau, cData, n, ws.workC)

	b1 = beta
	for i := range bsimo[:n] {
		bsimo[i] = 0
	}
	bsimo[0] = b1

	tauHrd := ws.tauHrd
	if n > 1 {
		impl.Dgehrd(n, 0, n-1, aData, n, tauHrd, ws.workHrd, len(ws.workHrd))
		impl.Dormhr(blas.Right, blas.NoTrans, m, n, 0, n-1, aData, n, tauHrd, cData, n, ws.workOrm, len(ws.workOrm))
	}

	for j := 0; j < n; j++ {
		for i := j + 2; i < n; i++ {
			aData[i*n+j] = 0
		}
	}

	tolScan := math.Max(tol, float64(n)*eps()*math.Max(frobA, math.Abs(b1/scaleB)*scaleB))
	nobs = n
	for j := 0; j < n-1; j++ {
		subdiag := aData[(j+1)*n+j]
		if math.Abs(subdiag) <= tolScan {
			nobs = j + 1
			break
		}
	}

	if nobs == 0 {
		return 0, 0, scaleB
	}

	if scaleA != 1.0 {
		invSA := 1.0 / scaleA
		for j := 0; j < nobs; j++ {
			for i := 0; i < nobs; i++ {
				aData[j*n+i] *= invSA
			}
		}
	}
	b1 /= scaleB

	return nobs, b1, scaleB
}

func vPolyRecurrence(aData []float64, n, nobs int) []Poly {
	wPolys := make([]Poly, nobs+1)
	wPolys[0] = Poly{1}
	mulBuf := make(Poly, 0, nobs+2)
	addBuf := make(Poly, 0, nobs+2)
	scaleBuf := make(Poly, 0, nobs+1)
	linPoly := Poly{1, 0}

	for k := 1; k <= nobs; k++ {
		r := nobs - k
		linPoly[1] = -aData[r*n+r]
		mulBuf = linPoly.MulTo(mulBuf, wPolys[k-1])
		for i := 1; i < k; i++ {
			hij := aData[r*n+r+i]
			if hij != 0 {
				scaleBuf = wPolys[k-1-i].ScaleTo(scaleBuf, -hij)
				addBuf, mulBuf = mulBuf, mulBuf.AddTo(addBuf, scaleBuf)
			}
		}
		wPolys[k] = make(Poly, len(mulBuf))
		copy(wPolys[k], mulBuf)
	}

	return wPolys
}

func computeNumerators(cData []float64, n, nobs, m int, scale []float64, wPolys []Poly) [][]float64 {
	nums := make([][]float64, m)
	scaleBuf := make(Poly, 0, nobs+1)
	addBuf := make(Poly, 0, nobs+2)
	for j := 0; j < m; j++ {
		var num Poly
		for k := 0; k < nobs; k++ {
			coeff := cData[j*n+k] * scale[k]
			if coeff != 0 {
				scaleBuf = wPolys[nobs-1-k].ScaleTo(scaleBuf, coeff)
				if len(num) == 0 {
					num = make(Poly, len(scaleBuf))
					copy(num, scaleBuf)
				} else {
					addBuf, num = num, num.AddTo(addBuf, scaleBuf)
				}
			}
		}
		if len(num) == 0 {
			nums[j] = []float64{0}
		} else {
			nums[j] = []float64(num)
		}
	}
	return nums
}

func ssToTFRow(Ac, Bc, Cc *mat.Dense, row, ncont, m int, obsTol float64, ws *tfWorkspace) (int, []float64, [][]float64) {
	zeroReturn := func() (int, []float64, [][]float64) {
		den := []float64{1}
		nums := make([][]float64, m)
		for j := 0; j < m; j++ {
			nums[j] = []float64{0}
		}
		return 0, den, nums
	}

	if ncont == 0 {
		return zeroReturn()
	}

	n := ncont

	formDualSIMO(Ac, Bc, Cc, row, n, m, ws)

	nobs, b1, _ := observabilityReduction(n, m, ws, obsTol)
	if nobs == 0 {
		return zeroReturn()
	}

	aData := ws.aData

	scale := make([]float64, nobs)
	scale[0] = b1
	for k := 1; k < nobs; k++ {
		scale[k] = scale[k-1] * aData[k*n+(k-1)]
	}

	// Scale upper triangle so V-poly recurrence works on monic form
	for i := 0; i < nobs; i++ {
		for j := i + 1; j < nobs; j++ {
			aData[i*n+j] *= scale[j] / scale[i]
		}
	}

	wPolys := vPolyRecurrence(aData, n, nobs)
	den := []float64(wPolys[nobs])
	nums := computeNumerators(ws.cData, n, nobs, m, scale, wPolys)

	return nobs, den, nums
}

type StateSpaceOpts struct {
	MinimalTol float64
}

type StateSpaceResult struct {
	Sys          *System
	MinimalOrder int
	BlockSizes   []int
}

func (tf *TransferFunc) StateSpace(opts *StateSpaceOpts) (*StateSpaceResult, error) {
	p, m := tf.Dims()

	if p == 0 || m == 0 {
		sys, _ := NewGain(&mat.Dense{}, tf.Dt)
		return &StateSpaceResult{Sys: sys, MinimalOrder: 0}, nil
	}

	degrees := make([]int, p)
	totalN := 0
	for i := 0; i < p; i++ {
		degrees[i] = len(tf.Den[i]) - 1
		totalN += degrees[i]
	}

	if totalN == 0 {
		D := mat.NewDense(p, m, nil)
		for i := 0; i < p; i++ {
			scale := 1.0 / tf.Den[i][0]
			for j := 0; j < m; j++ {
				D.Set(i, j, scale*tf.Num[i][j][0])
			}
		}
		sys, _ := NewGain(D, tf.Dt)
		return &StateSpaceResult{Sys: sys, MinimalOrder: 0, BlockSizes: degrees}, nil
	}

	smlnum := math.SmallestNonzeroFloat64 / eps()
	bignum := 1.0 / smlnum

	A := mat.NewDense(totalN, totalN, nil)
	B := mat.NewDense(totalN, m, nil)
	C := mat.NewDense(p, totalN, nil)
	D := mat.NewDense(p, m, nil)

	ja := 0
	for i := 0; i < p; i++ {
		di := degrees[i]
		if di == 0 {
			scale := 1.0 / tf.Den[i][0]
			for j := 0; j < m; j++ {
				D.Set(i, j, scale*tf.Num[i][j][0])
			}
			continue
		}

		leading := tf.Den[i][0]

		if math.Abs(leading) < smlnum {
			return nil, fmt.Errorf("row %d: %w", i, ErrSingularDenom)
		}

		umax := 0.0
		for j := 0; j < m; j++ {
			if v := math.Abs(tf.Num[i][j][0]); v > umax {
				umax = v
			}
		}

		if math.Abs(leading) < 1 && umax > math.Abs(leading)*bignum {
			return nil, fmt.Errorf("row %d: %w", i, ErrOverflow)
		}

		if di >= 1 {
			dmx := 0.0
			for k := 1; k <= di; k++ {
				if v := math.Abs(tf.Den[i][k]); v > dmx {
					dmx = v
				}
			}
			if math.Abs(leading) >= 1 {
				if umax > 1 && (dmx/math.Abs(leading)) > (bignum/umax) {
					return nil, fmt.Errorf("row %d: %w", i, ErrOverflow)
				}
			} else {
				if umax > 1 && dmx > (bignum*math.Abs(leading))/umax {
					return nil, fmt.Errorf("row %d: %w", i, ErrOverflow)
				}
			}
		}

		scale := 1.0 / leading

		for k := 0; k < di-1; k++ {
			A.Set(ja+k+1, ja+k, 1)
		}

		// Pad numerator to di+1 coefficients (left-pad with zeros)
		padNum := make([][]float64, m)
		for j := 0; j < m; j++ {
			padNum[j] = make([]float64, di+1)
			nj := len(tf.Num[i][j])
			off := di + 1 - nj
			for idx := 0; idx < nj; idx++ {
				if off+idx >= 0 {
					padNum[j][off+idx] = tf.Num[i][j][idx]
				}
			}
		}

		for k := 0; k < di; k++ {
			row := ja + di - 1 - k
			temp := -scale * tf.Den[i][k+1]
			A.Set(row, ja+di-1, temp)
			for j := 0; j < m; j++ {
				B.Set(row, j, padNum[j][k+1]+temp*padNum[j][0])
			}
		}

		if ja+di < totalN {
			A.Set(ja+di, ja+di-1, 0)
		}

		for j := 0; j < m; j++ {
			D.Set(i, j, scale*padNum[j][0])
		}

		// C row
		C.Set(i, ja+di-1, scale)

		ja += di
	}

	sys, err := newNoCopy(A, B, C, D, tf.Dt)
	if err != nil {
		return nil, err
	}

	if tf.Delay != nil {
		sys.Delay = slice2DToDense(tf.Delay)
	}

	return &StateSpaceResult{
		Sys:          sys,
		MinimalOrder: totalN,
		BlockSizes:   degrees,
	}, nil
}
