package controlsys

import (
	"fmt"
	"maps"
	"math"
)

type SystuneOptions struct {
	GridPoints     int
	MaxEvaluations int
}

type SystuneResult struct {
	Method     string
	Pass       bool
	Score      float64
	Iterations int
	Parameters map[string]float64
	Controller *System
	ClosedLoop *System
	Goals      []TuningGoalResult
}

func Systune(model *GeneralizedClosedLoop, goals []TuningGoal, opts *SystuneOptions) (*SystuneResult, error) {
	return GridTune(model, goals, opts)
}

func Looptune(model *GeneralizedClosedLoop, goals []TuningGoal, opts *SystuneOptions) (*SystuneResult, error) {
	return GridTune(model, goals, opts)
}

// GridTune searches a bounded Cartesian grid of the controller's free parameters.
func GridTune(model *GeneralizedClosedLoop, goals []TuningGoal, opts *SystuneOptions) (*SystuneResult, error) {
	if model == nil {
		return nil, fmt.Errorf("GridTune: nil model: %w", ErrDimensionMismatch)
	}
	controller := model.tunableController
	if controller == nil {
		return nil, fmt.Errorf("GridTune: controller is not tunable: %w", ErrDimensionMismatch)
	}
	if len(goals) == 0 {
		return nil, fmt.Errorf("GridTune: no goals: %w", ErrDimensionMismatch)
	}
	gridPoints := 5
	maxEvaluations := 100_000
	if opts != nil && opts.GridPoints > 0 {
		gridPoints = opts.GridPoints
	}
	if opts != nil && opts.MaxEvaluations > 0 {
		maxEvaluations = opts.MaxEvaluations
	}
	params := controller.FreeParameters()
	if len(params) == 0 {
		return nil, fmt.Errorf("GridTune: no free tunable parameters: %w", ErrDimensionMismatch)
	}
	evaluationCount := 1
	for _, param := range params {
		count := len(parameterGrid(param, gridPoints))
		if evaluationCount > maxEvaluations/count {
			return nil, fmt.Errorf("GridTune: Cartesian grid exceeds %d evaluations: %w", maxEvaluations, ErrDimensionMismatch)
		}
		evaluationCount *= count
	}

	best := &SystuneResult{Method: "cartesian-grid", Score: math.Inf(1)}
	iterations := 0
	values := make(map[string]float64, len(params))
	var search func(int) error
	search = func(idx int) error {
		if idx == len(params) {
			iterations++
			sampled, err := controller.SampleBlock(values)
			if err != nil {
				return err
			}
			candidate := *model
			candidate.controller = sampled
			candidate.tunableController = sampled
			closed, err := candidate.ClosedLoop(candidate.primaryAnalysisPointName())
			if err != nil {
				return err
			}
			goalResults, score, pass, err := evaluateTuningGoals(&candidate, closed, goals)
			if err != nil {
				return err
			}
			if score < best.Score {
				ctrl, err := sampled.CurrentSystem()
				if err != nil {
					return err
				}
				best = &SystuneResult{
					Method:     "cartesian-grid",
					Pass:       pass,
					Score:      score,
					Parameters: copyStringFloatMap(values),
					Controller: ctrl,
					ClosedLoop: closed,
					Goals:      goalResults,
				}
			}
			return nil
		}
		param := params[idx]
		candidates := parameterGrid(param, gridPoints)
		for _, value := range candidates {
			values[param.Name()] = value
			if err := search(idx + 1); err != nil {
				return err
			}
		}
		return nil
	}
	if err := search(0); err != nil {
		return nil, err
	}
	best.Iterations = iterations
	return best, nil
}

func evaluateTuningGoals(model *GeneralizedClosedLoop, primaryClosedLoop *System, goals []TuningGoal) ([]TuningGoalResult, float64, bool, error) {
	results := make([]TuningGoalResult, len(goals))
	score := 0.0
	pass := true
	primary := model.primaryAnalysisPointName()
	cache := map[tuningGoalResponseKey]*System{
		{point: primary, response: tuningGoalClosedLoopResponse}: primaryClosedLoop,
	}
	for i, goal := range goals {
		point := goal.spec.AnalysisPoint
		if point == "" {
			point = primary
		}
		key := tuningGoalResponseKey{point: point, response: tuningGoalResponseForType(goal.spec.Type)}
		sys := cache[key]
		if sys == nil {
			var err error
			sys, err = tuningGoalSystem(model, goal.spec)
			if err != nil {
				return nil, 0, false, err
			}
			cache[key] = sys
		}
		result, err := goal.evaluateSystem(sys)
		if err != nil {
			return nil, 0, false, err
		}
		results[i] = result
		if !result.Pass {
			pass = false
		}
		score += goalViolation(result)
	}
	return results, score, pass, nil
}

type tuningGoalResponseKey struct {
	point    string
	response tuningGoalResponse
}

func goalViolation(result TuningGoalResult) float64 {
	return result.Violation
}

func uniqueFreeTunableReals(params [][]*TunableReal) []*TunableReal {
	seen := make(map[string]bool)
	var out []*TunableReal
	for _, row := range params {
		for _, param := range row {
			if param == nil || param.Fixed() || seen[param.Name()] {
				continue
			}
			seen[param.Name()] = true
			out = append(out, param)
		}
	}
	return out
}

func parameterGrid(param *TunableReal, points int) []float64 {
	bounds := param.Bounds()
	if points < 2 || (bounds.Lower == 0 && bounds.Upper == 0) {
		return []float64{param.Value()}
	}
	out := make([]float64, points)
	for i := range out {
		out[i] = bounds.Lower + float64(i)*(bounds.Upper-bounds.Lower)/float64(points-1)
	}
	return out
}

func copyStringFloatMap(src map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(src))
	maps.Copy(out, src)
	return out
}
