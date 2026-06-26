package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
)

type TuningGoalType int

const (
	TuningGoalTracking TuningGoalType = iota
	TuningGoalRejection
	TuningGoalSensitivity
	TuningGoalWeightedGain
	TuningGoalLoopShape
	TuningGoalMargin
	TuningGoalPole
	TuningGoalOvershoot
)

type TuningGoalSpec struct {
	Name string
	Type TuningGoalType
	Max  float64
	Min  float64
}

type TuningGoal struct {
	spec TuningGoalSpec
}

type TuningGoalResult struct {
	GoalName    string
	Pass        bool
	Value       float64
	Limit       float64
	Diagnostics map[string]float64
}

func NewTuningGoal(spec TuningGoalSpec) (TuningGoal, error) {
	if spec.Name == "" {
		return TuningGoal{}, fmt.Errorf("NewTuningGoal: name is empty: %w", ErrDimensionMismatch)
	}
	if spec.Type != TuningGoalPole && (spec.Max < 0 || spec.Min < 0) {
		return TuningGoal{}, fmt.Errorf("NewTuningGoal: negative bound: %w", ErrDimensionMismatch)
	}
	return TuningGoal{spec: spec}, nil
}

// NewTrackingGoal returns a tracking goal and panics if the goal is invalid.
// Use NewTuningGoal for error-returning construction.
func NewTrackingGoal(name string, maxError float64) TuningGoal {
	return mustTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalTracking, Max: maxError})
}

// NewRejectionGoal returns a rejection goal and panics if the goal is invalid.
// Use NewTuningGoal for error-returning construction.
func NewRejectionGoal(name string, maxGain float64) TuningGoal {
	return mustTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalRejection, Max: maxGain})
}

// NewSensitivityGoal returns a sensitivity goal and panics if the goal is invalid.
// Use NewTuningGoal for error-returning construction.
func NewSensitivityGoal(name string, maxGain float64) TuningGoal {
	return mustTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalSensitivity, Max: maxGain})
}

// NewWeightedGainGoal returns a weighted-gain goal and panics if the goal is invalid.
// Use NewTuningGoal for error-returning construction.
func NewWeightedGainGoal(name string, maxGain float64) TuningGoal {
	return mustTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalWeightedGain, Max: maxGain})
}

// NewLoopShapeGoal returns a loop-shape goal and panics if the goal is invalid.
// Use NewTuningGoal for error-returning construction.
func NewLoopShapeGoal(name string, minGain, maxGain float64) TuningGoal {
	return mustTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalLoopShape, Min: minGain, Max: maxGain})
}

// NewMarginGoal returns a stability-margin goal and panics if the goal is invalid.
// Use NewTuningGoal for error-returning construction.
func NewMarginGoal(name string, minGainMarginDB, minPhaseMarginDeg float64) TuningGoal {
	return mustTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalMargin, Min: minGainMarginDB, Max: minPhaseMarginDeg})
}

// NewPoleGoal returns a pole-location goal and panics if the goal is invalid.
// Use NewTuningGoal for error-returning construction.
func NewPoleGoal(name string, maxRealPart float64) TuningGoal {
	return mustTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalPole, Max: maxRealPart})
}

// NewOvershootGoal returns an overshoot goal and panics if the goal is invalid.
// Use NewTuningGoal for error-returning construction.
func NewOvershootGoal(name string, maxPercent float64) TuningGoal {
	return mustTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalOvershoot, Max: maxPercent})
}

func mustTuningGoal(spec TuningGoalSpec) TuningGoal {
	goal, err := NewTuningGoal(spec)
	if err != nil {
		panic(err)
	}
	return goal
}

func (g TuningGoal) Name() string {
	return g.spec.Name
}

func (g TuningGoal) Evaluate(model any) (TuningGoalResult, error) {
	sys, err := tuningGoalSystem(model)
	if err != nil {
		return TuningGoalResult{}, err
	}
	switch g.spec.Type {
	case TuningGoalTracking:
		return g.evaluateTracking(sys)
	case TuningGoalRejection, TuningGoalSensitivity, TuningGoalWeightedGain:
		return g.evaluateMaxGain(sys)
	case TuningGoalLoopShape:
		return g.evaluateLoopShape(sys)
	case TuningGoalMargin:
		return g.evaluateMargin(sys)
	case TuningGoalPole:
		return g.evaluatePole(sys)
	case TuningGoalOvershoot:
		return g.evaluateOvershoot(sys)
	default:
		return TuningGoalResult{}, fmt.Errorf("TuningGoal.Evaluate: unsupported goal type: %w", ErrDimensionMismatch)
	}
}

func tuningGoalSystem(model any) (*System, error) {
	switch v := model.(type) {
	case *System:
		return v.Copy(), nil
	case *GeneralizedModel:
		return v.CurrentSystem()
	case *GeneralizedClosedLoop:
		return v.ClosedLoop(v.primaryAnalysisPointName())
	default:
		return nil, fmt.Errorf("TuningGoal.Evaluate: unsupported model %T: %w", model, ErrDimensionMismatch)
	}
}

func firstAnalysisPointName(points map[string]AnalysisPoint) string {
	for name := range points {
		return name
	}
	return ""
}

func (g TuningGoal) evaluateTracking(sys *System) (TuningGoalResult, error) {
	dc, err := sys.DCGain()
	if err != nil {
		return TuningGoalResult{}, err
	}
	errVal := maxDCErrorFromOne(dc)
	return g.scalarResult(errVal, g.spec.Max, errVal <= g.spec.Max, map[string]float64{"dc_error": errVal}), nil
}

func (g TuningGoal) evaluateMaxGain(sys *System) (TuningGoalResult, error) {
	value, err := maxFrequencyGain(sys, nil)
	if err != nil {
		return TuningGoalResult{}, err
	}
	return g.scalarResult(value, g.spec.Max, value <= g.spec.Max, map[string]float64{"max_gain": value}), nil
}

func (g TuningGoal) evaluateLoopShape(sys *System) (TuningGoalResult, error) {
	value, err := maxFrequencyGain(sys, nil)
	if err != nil {
		return TuningGoalResult{}, err
	}
	pass := value >= g.spec.Min && value <= g.spec.Max
	return g.scalarResult(value, g.spec.Max, pass, map[string]float64{"max_gain": value, "min_gain": g.spec.Min}), nil
}

func (g TuningGoal) evaluateMargin(sys *System) (TuningGoalResult, error) {
	margin, err := Margin(sys)
	if err != nil {
		return TuningGoalResult{}, err
	}
	pass := margin.GainMargin >= g.spec.Min && margin.PhaseMargin >= g.spec.Max
	diag := map[string]float64{"gain_margin_db": margin.GainMargin, "phase_margin_deg": margin.PhaseMargin}
	return g.scalarResult(math.Min(margin.GainMargin, margin.PhaseMargin), g.spec.Max, pass, diag), nil
}

func (g TuningGoal) evaluatePole(sys *System) (TuningGoalResult, error) {
	poles, err := sys.Poles()
	if err != nil {
		return TuningGoalResult{}, err
	}
	maxReal := math.Inf(-1)
	for _, p := range poles {
		if real(p) > maxReal {
			maxReal = real(p)
		}
	}
	if len(poles) == 0 {
		maxReal = math.Inf(-1)
	}
	return g.scalarResult(maxReal, g.spec.Max, maxReal <= g.spec.Max, map[string]float64{"max_real_pole": maxReal}), nil
}

func (g TuningGoal) evaluateOvershoot(sys *System) (TuningGoalResult, error) {
	info, err := StepInfoForSystem(sys, 0, nil)
	if err != nil {
		return TuningGoalResult{}, err
	}
	maxOvershoot := 0.0
	for _, metric := range info.Metrics {
		if metric.Overshoot > maxOvershoot {
			maxOvershoot = metric.Overshoot
		}
	}
	return g.scalarResult(maxOvershoot, g.spec.Max, maxOvershoot <= g.spec.Max, map[string]float64{"overshoot_percent": maxOvershoot}), nil
}

func (g TuningGoal) scalarResult(value, limit float64, pass bool, diag map[string]float64) TuningGoalResult {
	return TuningGoalResult{GoalName: g.spec.Name, Pass: pass, Value: value, Limit: limit, Diagnostics: diag}
}

func maxDCErrorFromOne(dc interface {
	Dims() (int, int)
	At(int, int) float64
}) float64 {
	r, c := dc.Dims()
	maxErr := 0.0
	for i := range r {
		for j := range c {
			want := 0.0
			if i == j {
				want = 1
			}
			if err := math.Abs(dc.At(i, j) - want); err > maxErr {
				maxErr = err
			}
		}
	}
	return maxErr
}

func maxFrequencyGain(sys *System, omega []float64) (float64, error) {
	if omega == nil {
		omega = logspace(-2, 2, 80)
	}
	resp, err := sys.FreqResponse(omega)
	if err != nil {
		return 0, err
	}
	maxGain := 0.0
	for k := range omega {
		for i := 0; i < resp.P; i++ {
			for j := 0; j < resp.M; j++ {
				if gain := cmplx.Abs(resp.At(k, i, j)); gain > maxGain {
					maxGain = gain
				}
			}
		}
	}
	return maxGain, nil
}
