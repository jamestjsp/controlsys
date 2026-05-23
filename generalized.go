package controlsys

import "fmt"

type NumericBlock interface {
	CurrentSystem() (*System, error)
}

type fixedSystemBlock struct {
	sys *System
}

func (b fixedSystemBlock) CurrentSystem() (*System, error) {
	if b.sys == nil {
		return nil, fmt.Errorf("fixed system block is nil: %w", ErrDimensionMismatch)
	}
	return b.sys.Copy(), nil
}

type GeneralizedModel struct {
	name           string
	block          NumericBlock
	inputName      []string
	outputName     []string
	analysisPoints map[string]AnalysisPoint
}

type AnalysisPoint struct {
	Name string
}

func NewGeneralizedModel(name string, block any) (*GeneralizedModel, error) {
	if name == "" {
		return nil, fmt.Errorf("NewGeneralizedModel: name is empty: %w", ErrDimensionMismatch)
	}
	numeric, err := numericBlockFromAny(block)
	if err != nil {
		return nil, err
	}
	return &GeneralizedModel{name: name, block: numeric, analysisPoints: make(map[string]AnalysisPoint)}, nil
}

func numericBlockFromAny(block any) (NumericBlock, error) {
	switch b := block.(type) {
	case NumericBlock:
		return b, nil
	case *System:
		return fixedSystemBlock{sys: b}, nil
	default:
		return nil, fmt.Errorf("unsupported generalized block %T: %w", block, ErrDimensionMismatch)
	}
}

func (g *GeneralizedModel) SetInputName(names ...string) {
	g.inputName = copyStringSlice(names)
}

func (g *GeneralizedModel) SetOutputName(names ...string) {
	g.outputName = copyStringSlice(names)
}

func (g *GeneralizedModel) InsertAnalysisPoint(name string) {
	if g.analysisPoints == nil {
		g.analysisPoints = make(map[string]AnalysisPoint)
	}
	g.analysisPoints[name] = AnalysisPoint{Name: name}
}

func (g *GeneralizedModel) HasAnalysisPoint(name string) bool {
	if g == nil {
		return false
	}
	_, ok := g.analysisPoints[name]
	return ok
}

func (g *GeneralizedModel) AnalysisPoint(name string) (AnalysisPoint, error) {
	if g == nil {
		return AnalysisPoint{}, fmt.Errorf("GeneralizedModel.AnalysisPoint: nil model: %w", ErrDimensionMismatch)
	}
	ap, ok := g.analysisPoints[name]
	if !ok {
		return AnalysisPoint{}, fmt.Errorf("%q: %w", name, ErrSignalNotFound)
	}
	return ap, nil
}

func (g *GeneralizedModel) CurrentSystem() (*System, error) {
	if g == nil || g.block == nil {
		return nil, fmt.Errorf("GeneralizedModel.CurrentSystem: nil model: %w", ErrDimensionMismatch)
	}
	sys, err := g.block.CurrentSystem()
	if err != nil {
		return nil, err
	}
	if g.inputName != nil {
		sys.InputName = copyStringSlice(g.inputName)
	}
	if g.outputName != nil {
		sys.OutputName = copyStringSlice(g.outputName)
	}
	return sys, nil
}

type GeneralizedClosedLoop struct {
	name           string
	plant          *System
	controller     NumericBlock
	analysisPoints map[string]AnalysisPoint
}

func NewGeneralizedClosedLoop(name string, plant *System, controller any, analysisPoint string) (*GeneralizedClosedLoop, error) {
	ctrl, err := numericBlockFromAny(controller)
	if err != nil {
		return nil, err
	}
	if plant == nil {
		return nil, fmt.Errorf("NewGeneralizedClosedLoop: nil plant: %w", ErrDimensionMismatch)
	}
	g := &GeneralizedClosedLoop{
		name:           name,
		plant:          plant.Copy(),
		controller:     ctrl,
		analysisPoints: make(map[string]AnalysisPoint),
	}
	g.analysisPoints[analysisPoint] = AnalysisPoint{Name: analysisPoint}
	return g, nil
}

func (g *GeneralizedClosedLoop) AnalysisPoint(name string) (AnalysisPoint, error) {
	if g == nil {
		return AnalysisPoint{}, fmt.Errorf("GeneralizedClosedLoop.AnalysisPoint: nil model: %w", ErrDimensionMismatch)
	}
	ap, ok := g.analysisPoints[name]
	if !ok {
		return AnalysisPoint{}, fmt.Errorf("%q: %w", name, ErrSignalNotFound)
	}
	return ap, nil
}

func (g *GeneralizedClosedLoop) OpenLoop(name string) (*System, error) {
	if _, err := g.AnalysisPoint(name); err != nil {
		return nil, err
	}
	controller, err := g.controller.CurrentSystem()
	if err != nil {
		return nil, err
	}
	return Series(controller, g.plant)
}

func (g *GeneralizedClosedLoop) ClosedLoop(name string) (*System, error) {
	return g.ComplementarySensitivity(name)
}

func (g *GeneralizedClosedLoop) ComplementarySensitivity(name string) (*System, error) {
	L, err := g.OpenLoop(name)
	if err != nil {
		return nil, err
	}
	return Feedback(L, nil, -1)
}

func (g *GeneralizedClosedLoop) Sensitivity(name string) (*System, error) {
	T, err := g.ComplementarySensitivity(name)
	if err != nil {
		return nil, err
	}
	one, err := NewGain(eyeDense(1), T.Dt)
	if err != nil {
		return nil, err
	}
	negT := T.Copy()
	negT.C.Scale(-1, negT.C)
	negT.D.Scale(-1, negT.D)
	return Parallel(one, negT)
}
