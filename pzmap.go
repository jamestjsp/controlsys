package controlsys

type PzmapResult struct {
	Poles []complex128
	Zeros []complex128
}

func Pzmap(sys *System) (*PzmapResult, error) {
	poles, err := sys.Poles()
	if err != nil {
		return nil, err
	}
	zeros, err := sys.Zeros()
	if err != nil {
		return nil, err
	}
	return &PzmapResult{Poles: poles, Zeros: zeros}, nil
}
