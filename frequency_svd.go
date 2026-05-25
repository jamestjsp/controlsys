package controlsys

import (
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

type complexSVDWorkspace struct {
	block  []float64
	eig    []float64
	work   []float64
	nSV    int
	gramN  int
	blockN int
	useCol bool
}

func newComplexSVDWorkspace(p, m int) *complexSVDWorkspace {
	gramN := min(p, m)
	blockN := 2 * gramN
	block := make([]float64, blockN*blockN)
	eig := make([]float64, blockN)
	workQuery := make([]float64, 1)
	impl.Dsyev(lapack.EVNone, blas.Upper, blockN, block, blockN, eig, workQuery, -1)
	work := make([]float64, int(workQuery[0]))
	return &complexSVDWorkspace{
		block:  block,
		eig:    eig,
		work:   work,
		nSV:    gramN,
		gramN:  gramN,
		blockN: blockN,
		useCol: m <= p,
	}
}

func (ws *complexSVDWorkspace) singularValuesFromFlat(dst []float64, data []complex128, base, p, m int) {
	ws.fillBlock(func(i, j int) complex128 {
		return data[base+i*m+j]
	}, p, m)
	ws.singularValues(dst)
}

func (ws *complexSVDWorkspace) singularValuesFromNested(dst []float64, data [][]complex128, p, m int) {
	ws.fillBlock(func(i, j int) complex128 {
		return data[i][j]
	}, p, m)
	ws.singularValues(dst)
}

func (ws *complexSVDWorkspace) fillBlock(at func(i, j int) complex128, p, m int) {
	n := ws.gramN
	stride := ws.blockN
	for i := range ws.block {
		ws.block[i] = 0
	}

	for a := range n {
		for b := range n {
			var g complex128
			if ws.useCol {
				for row := range p {
					g += cmplx.Conj(at(row, a)) * at(row, b)
				}
			} else {
				for col := range m {
					g += at(a, col) * cmplx.Conj(at(b, col))
				}
			}

			re, im := real(g), imag(g)
			ws.block[a*stride+b] = re
			ws.block[a*stride+n+b] = -im
			ws.block[(n+a)*stride+b] = im
			ws.block[(n+a)*stride+n+b] = re
		}
	}
}

func (ws *complexSVDWorkspace) singularValues(dst []float64) {
	ok := impl.Dsyev(lapack.EVNone, blas.Upper, ws.blockN, ws.block, ws.blockN, ws.eig, ws.work, len(ws.work))
	if !ok {
		for i := range dst {
			dst[i] = math.NaN()
		}
		return
	}
	scale := 1.0
	for _, lambda := range ws.eig {
		scale = max(scale, math.Abs(lambda))
	}
	for i := range dst {
		lambda := nonnegativeGramEigenvalue(ws.eig[ws.blockN-1-2*i], scale)
		dst[i] = math.Sqrt(lambda)
	}
}

func nonnegativeGramEigenvalue(lambda, scale float64) float64 {
	if lambda >= 0 {
		return lambda
	}
	if lambda > -1e-12*max(1, scale) {
		return 0
	}
	return lambda
}
