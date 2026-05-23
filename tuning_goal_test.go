package controlsys

import "testing"

func TestTuningGoalsEvaluatePassFailFamilies(t *testing.T) {
	sys := makeSISO(-2, 2, 1, 0)
	goals := []TuningGoal{
		NewTrackingGoal("track", 1.2),
		NewRejectionGoal("reject", 1.2),
		NewSensitivityGoal("sens", 1.2),
		NewWeightedGainGoal("gain", 1.2),
		NewLoopShapeGoal("loop", 0.5, 2.0),
		NewMarginGoal("margin", 0, 0),
		NewPoleGoal("poles", 0),
		NewOvershootGoal("overshoot", 5),
	}
	for _, goal := range goals {
		result, err := goal.Evaluate(sys)
		if err != nil {
			t.Fatalf("%s Evaluate: %v", goal.Name(), err)
		}
		if result.GoalName != goal.Name() || len(result.Diagnostics) == 0 {
			t.Fatalf("%s result missing diagnostics: %#v", goal.Name(), result)
		}
	}
}

func TestTuningGoalsKnownFailuresAndGeneralizedCurrentValue(t *testing.T) {
	sys := makeSISO(-1, 1, 2, 0)
	failGoal := NewWeightedGainGoal("too_small", 0.5)
	res, err := failGoal.Evaluate(sys)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res.Pass {
		t.Fatalf("expected weighted gain failure, got %#v", res)
	}

	k, _ := NewTunableReal("K", 0.25, TunableBounds{Lower: 0, Upper: 1})
	gm, err := NewGeneralizedModel("gain", NewTunableGain("Kblock", [][]*TunableReal{{k}}, 0))
	if err != nil {
		t.Fatal(err)
	}
	passGoal := NewWeightedGainGoal("small_gain", 0.5)
	gres, err := passGoal.Evaluate(gm)
	if err != nil {
		t.Fatalf("Evaluate generalized: %v", err)
	}
	if !gres.Pass {
		t.Fatalf("expected generalized gain pass, got %#v", gres)
	}
}

func TestTuningGoalValidation(t *testing.T) {
	if _, err := NewTuningGoal(TuningGoalSpec{Name: "", Type: TuningGoalWeightedGain, Max: 1}); err == nil {
		t.Fatal("empty goal name should fail")
	}
	if _, err := NewTuningGoal(TuningGoalSpec{Name: "bad", Type: TuningGoalWeightedGain, Max: -1}); err == nil {
		t.Fatal("negative max should fail")
	}
	if _, err := NewTuningGoal(TuningGoalSpec{Name: "pole_spec", Type: TuningGoalPole, Min: -2, Max: -0.5}); err != nil {
		t.Fatalf("negative pole bounds should pass: %v", err)
	}
	goal := NewPoleGoal("stable_fast", -0.5)
	pass, err := goal.Evaluate(makeSISO(-1, 1, 1, 0))
	if err != nil {
		t.Fatalf("Evaluate stable pole goal: %v", err)
	}
	if !pass.Pass {
		t.Fatalf("expected pole goal pass, got %#v", pass)
	}
	fail, err := goal.Evaluate(makeSISO(-0.2, 1, 1, 0))
	if err != nil {
		t.Fatalf("Evaluate slow pole goal: %v", err)
	}
	if fail.Pass {
		t.Fatalf("expected pole goal failure, got %#v", fail)
	}
}
