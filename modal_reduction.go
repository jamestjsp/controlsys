package controlsys

import (
	"math"
	"sort"
)

type ModalTruncateOptions struct {
	Order       int
	MaxRealPart float64
}

type ModalReductionResult struct {
	Sys    *System
	Order  int
	Method string
	Kept   []int
}

func ModalTruncate(sys *System, opts *ModalTruncateOptions) (*ModalReductionResult, error) {
	if opts == nil {
		opts = &ModalTruncateOptions{}
	}
	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard("ModalTruncate"); err != nil {
		return nil, err
	}
	if err := policy.requireDelayFree("ModalTruncate"); err != nil {
		return nil, err
	}
	n, _, _ := sys.Dims()
	if n == 0 {
		return &ModalReductionResult{Sys: sys.Copy(), Order: 0, Method: "modal-truncate"}, nil
	}
	order := opts.Order
	if order == 0 {
		order = modalAutoOrder(sys, opts.MaxRealPart)
	}
	if order < 1 || order > n {
		return nil, ErrInvalidOrder
	}
	if order == n {
		return &ModalReductionResult{Sys: sys.Copy(), Order: n, Method: "modal-truncate", Kept: rangeInts(n)}, nil
	}
	elim := make([]int, 0, n-order)
	for i := order; i < n; i++ {
		elim = append(elim, i)
	}
	reduced, err := Modred(sys, elim, Truncate)
	if err != nil {
		return nil, err
	}
	return &ModalReductionResult{Sys: reduced, Order: order, Method: "modal-truncate", Kept: rangeInts(order)}, nil
}

func modalAutoOrder(sys *System, maxRealPart float64) int {
	poles, err := sys.Poles()
	if err != nil || len(poles) == 0 {
		return 0
	}
	if maxRealPart == 0 {
		maxRealPart = math.Inf(1)
	}
	type poleScore struct {
		index int
		real  float64
	}
	scores := make([]poleScore, len(poles))
	for i, p := range poles {
		scores[i] = poleScore{index: i, real: real(p)}
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].real > scores[j].real })
	order := 0
	for _, score := range scores {
		if score.real >= maxRealPart {
			order++
		}
	}
	if order == 0 {
		order = len(poles) / 2
		if order == 0 {
			order = 1
		}
	}
	return order
}

func rangeInts(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}
