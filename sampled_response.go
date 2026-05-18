package controlsys

import "math/cmplx"

type sampledComplexResponse struct {
	data  []complex128
	omega []float64
	p, m  int
}

type sampledComplexGridResponse struct {
	response [][][]complex128
	omega    []float64
	p, m     int
}

func newSampledComplexResponse(data []complex128, omega []float64, p, m int) sampledComplexResponse {
	return sampledComplexResponse{data: data, omega: omega, p: p, m: m}
}

func (r sampledComplexResponse) offset(freq, output, input int) int {
	return freq*r.p*r.m + output*r.m + input
}

func (r sampledComplexResponse) at(freq, output, input int) complex128 {
	return r.data[r.offset(freq, output, input)]
}

func (r sampledComplexResponse) blockOffset(freq int) int {
	return freq * r.p * r.m
}

func (r sampledComplexResponse) singularValues(dst []float64, ws *complexSVDWorkspace, freq int) {
	if r.p == 1 && r.m == 1 {
		dst[0] = cmplx.Abs(r.data[freq])
		return
	}
	ws.singularValuesFromFlat(dst, r.data, r.blockOffset(freq), r.p, r.m)
}

func (r sampledComplexResponse) copyToGrid(dst [][][]complex128) {
	for k := range r.omega {
		base := r.blockOffset(k)
		for i := 0; i < r.p; i++ {
			copy(dst[k][i], r.data[base+i*r.m:base+(i+1)*r.m])
		}
	}
}

func newSampledComplexGridResponse(response [][][]complex128, omega []float64, p, m int) sampledComplexGridResponse {
	return sampledComplexGridResponse{response: response, omega: omega, p: p, m: m}
}

func (r sampledComplexGridResponse) at(freq, output, input int) complex128 {
	return r.response[freq][output][input]
}

func (r sampledComplexGridResponse) singularValues(dst []float64, ws *complexSVDWorkspace, freq int) {
	if r.p == 1 && r.m == 1 {
		dst[0] = cmplx.Abs(r.response[freq][0][0])
		return
	}
	ws.singularValuesFromNested(dst, r.response[freq], r.p, r.m)
}
