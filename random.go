package controlsys

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func Rss(n, p, m int) (*System, error) {
	if n < 0 || p < 1 || m < 1 {
		return nil, fmt.Errorf("controlsys: invalid dimensions n=%d p=%d m=%d", n, p, m)
	}
	return randomSS(n, p, m, 0, true)
}

func Drss(n, p, m int, dt float64) (*System, error) {
	if n < 0 || p < 1 || m < 1 {
		return nil, fmt.Errorf("controlsys: invalid dimensions n=%d p=%d m=%d", n, p, m)
	}
	if dt <= 0 {
		return nil, ErrInvalidSampleTime
	}
	return randomSS(n, p, m, dt, false)
}

func randomSS(n, p, m int, dt float64, continuous bool) (*System, error) {
	if n == 0 {
		dData := make([]float64, p*m)
		for i := range dData {
			dData[i] = rand.NormFloat64()
		}
		return NewGain(mat.NewDense(p, m, dData), dt)
	}

	A := randomStableA(n, continuous)

	bData := make([]float64, n*m)
	for i := range bData {
		bData[i] = rand.NormFloat64()
	}
	B := mat.NewDense(n, m, bData)

	cData := make([]float64, p*n)
	for i := range cData {
		cData[i] = rand.NormFloat64()
	}
	C := mat.NewDense(p, n, cData)

	D := mat.NewDense(p, m, nil)

	return newNoCopy(A, B, C, D, dt)
}

func randomStableA(n int, continuous bool) *mat.Dense {
	// Generate random eigenvalue structure with mix of real and complex pairs
	diagData := make([]float64, n*n)
	i := 0
	for i < n {
		if i+1 < n && rand.Float64() < 0.5 {
			if continuous {
				sigma := -rand.Float64()*4 - 0.1
				omega := rand.Float64()*4 + 0.1
				diagData[i*n+i] = sigma
				diagData[i*n+i+1] = omega
				diagData[(i+1)*n+i] = -omega
				diagData[(i+1)*n+i+1] = sigma
			} else {
				r := rand.Float64()*0.9 + 0.01
				theta := rand.Float64() * math.Pi
				diagData[i*n+i] = r * math.Cos(theta)
				diagData[i*n+i+1] = r * math.Sin(theta)
				diagData[(i+1)*n+i] = -r * math.Sin(theta)
				diagData[(i+1)*n+i+1] = r * math.Cos(theta)
			}
			i += 2
		} else {
			if continuous {
				diagData[i*n+i] = -rand.Float64()*4 - 0.1
			} else {
				diagData[i*n+i] = rand.Float64()*1.8 - 0.9
			}
			i++
		}
	}
	D := mat.NewDense(n, n, diagData)

	Q := randomOrthogonal(n)

	// A = Q * D * Q'
	tmp := mat.NewDense(n, n, nil)
	tmp.Mul(Q, D)
	A := mat.NewDense(n, n, nil)
	A.Mul(tmp, Q.T())
	return A
}

func randomOrthogonal(n int) *mat.Dense {
	data := make([]float64, n*n)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	M := mat.NewDense(n, n, data)
	var qr mat.QR
	qr.Factorize(M)
	Q := mat.NewDense(n, n, nil)
	qr.QTo(Q)
	return Q
}
