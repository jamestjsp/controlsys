package controlsys

import (
	"fmt"
	"math"
)

type SystuneOptions struct {
	GridPoints int
}

type SystuneResult struct {
	Pass       bool
	Score      float64
	Iterations int
	Parameters map[string]float64
	Controller *System
	ClosedLoop *System
	Goals      []TuningGoalResult
}

func Systune(model *GeneralizedClosedLoop, goals []TuningGoal, opts *SystuneOptions) (*SystuneResult, error) {
	return tuneFixedStructure(model, goals, opts)
}

func Looptune(model *GeneralizedClosedLoop, goals []TuningGoal, opts *SystuneOptions) (*SystuneResult, error) {
	return tuneFixedStructure(model, goals, opts)
}

func tuneFixedStructure(model *GeneralizedClosedLoop, goals []TuningGoal, opts *SystuneOptions) (*SystuneResult, error) {
	if model == nil {
		return nil, fmt.Errorf("Systune: nil model: %w", ErrDimensionMismatch)
	}
	gain, ok := model.controllerRaw.(*TunableGain)
	if !ok {
		return nil, fmt.Errorf("Systune: only TunableGain controllers are supported by this tracer: %w", ErrDimensionMismatch)
	}
	if len(goals) == 0 {
		return nil, fmt.Errorf("Systune: no goals: %w", ErrDimensionMismatch)
	}
	gridPoints := 5
	if opts != nil && opts.GridPoints > 0 {
		gridPoints = opts.GridPoints
	}
	params := uniqueFreeTunableReals(gain.D)
	if len(params) == 0 {
		return nil, fmt.Errorf("Systune: no free tunable parameters: %w", ErrDimensionMismatch)
	}

	best := &SystuneResult{Score: math.Inf(1)}
	iterations := 0
	values := make(map[string]float64, len(params))
	var search func(int) error
	search = func(idx int) error {
		if idx == len(params) {
			iterations++
			sampled, err := gain.Sample(values)
			if err != nil {
				return err
			}
			candidate := *model
			candidate.controller = sampled
			candidate.controllerRaw = sampled
			closed, err := candidate.ClosedLoop(firstAnalysisPointName(candidate.analysisPoints))
			if err != nil {
				return err
			}
			goalResults, score, pass, err := evaluateTuningGoals(closed, goals)
			if err != nil {
				return err
			}
			if score < best.Score {
				ctrl, err := sampled.CurrentSystem()
				if err != nil {
					return err
				}
				best = &SystuneResult{
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

func evaluateTuningGoals(sys *System, goals []TuningGoal) ([]TuningGoalResult, float64, bool, error) {
	results := make([]TuningGoalResult, len(goals))
	score := 0.0
	pass := true
	for i, goal := range goals {
		result, err := goal.Evaluate(sys)
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

func goalViolation(result TuningGoalResult) float64 {
	if result.Pass {
		return 0
	}
	if result.Limit == 0 {
		return math.Abs(result.Value)
	}
	return math.Abs((result.Value - result.Limit) / result.Limit)
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
	for k, v := range src {
		out[k] = v
	}
	return out
}
