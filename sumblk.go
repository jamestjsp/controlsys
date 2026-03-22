package controlsys

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"

	"gonum.org/v1/gonum/mat"
)

type tokenKind int

const (
	tokIdent tokenKind = iota
	tokNumber
	tokEquals
	tokPlus
	tokMinus
	tokStar
	tokSemicolon
	tokEOF
)

type token struct {
	kind tokenKind
	text string
	num  float64
}

type signalTerm struct {
	name  string
	coeff float64
}

type equation struct {
	output string
	terms  []signalTerm
}

func tokenize(expr string) ([]token, error) {
	var tokens []token
	runes := []rune(expr)
	i := 0
	for i < len(runes) {
		r := runes[i]
		if unicode.IsSpace(r) {
			i++
			continue
		}
		switch r {
		case '=':
			tokens = append(tokens, token{kind: tokEquals, text: "="})
			i++
		case '+':
			tokens = append(tokens, token{kind: tokPlus, text: "+"})
			i++
		case '-':
			tokens = append(tokens, token{kind: tokMinus, text: "-"})
			i++
		case '*':
			tokens = append(tokens, token{kind: tokStar, text: "*"})
			i++
		case ';':
			tokens = append(tokens, token{kind: tokSemicolon, text: ";"})
			i++
		default:
			if unicode.IsLetter(r) || r == '_' {
				start := i
				for i < len(runes) && (unicode.IsLetter(runes[i]) || unicode.IsDigit(runes[i]) || runes[i] == '_') {
					i++
				}
				tokens = append(tokens, token{kind: tokIdent, text: string(runes[start:i])})
			} else if unicode.IsDigit(r) || r == '.' {
				start := i
				for i < len(runes) && (unicode.IsDigit(runes[i]) || runes[i] == '.') {
					i++
				}
				s := string(runes[start:i])
				n, err := strconv.ParseFloat(s, 64)
				if err != nil {
					return nil, fmt.Errorf("sumblk: invalid number %q: %w", s, ErrInvalidExpression)
				}
				tokens = append(tokens, token{kind: tokNumber, text: s, num: n})
			} else {
				return nil, fmt.Errorf("sumblk: unexpected character %q: %w", string(r), ErrInvalidExpression)
			}
		}
	}
	tokens = append(tokens, token{kind: tokEOF})
	return tokens, nil
}

func parseEquations(tokens []token) ([]equation, error) {
	pos := 0

	peek := func() token { return tokens[pos] }
	advance := func() token {
		t := tokens[pos]
		pos++
		return t
	}
	expect := func(k tokenKind) (token, error) {
		t := peek()
		if t.kind != k {
			return t, fmt.Errorf("sumblk: expected %d, got %q: %w", k, t.text, ErrInvalidExpression)
		}
		advance()
		return t, nil
	}

	parseTerm := func(isFirst bool) (signalTerm, error) {
		coeff := 1.0
		neg := false

		if isFirst {
			if peek().kind == tokMinus {
				advance()
				neg = true
			}
		}

		if peek().kind == tokNumber {
			t := advance()
			if _, err := expect(tokStar); err != nil {
				return signalTerm{}, fmt.Errorf("sumblk: expected '*' after number: %w", ErrInvalidExpression)
			}
			coeff = t.num
		}

		ident, err := expect(tokIdent)
		if err != nil {
			return signalTerm{}, fmt.Errorf("sumblk: expected signal name: %w", ErrInvalidExpression)
		}

		if neg {
			coeff = -coeff
		}
		return signalTerm{name: ident.text, coeff: coeff}, nil
	}

	parseEq := func() (equation, error) {
		out, err := expect(tokIdent)
		if err != nil {
			return equation{}, fmt.Errorf("sumblk: expected output name: %w", ErrInvalidExpression)
		}
		if _, err := expect(tokEquals); err != nil {
			return equation{}, err
		}

		first, err := parseTerm(true)
		if err != nil {
			return equation{}, err
		}
		terms := []signalTerm{first}

		for peek().kind == tokPlus || peek().kind == tokMinus {
			op := advance()
			t, err := parseTerm(false)
			if err != nil {
				return equation{}, err
			}
			if op.kind == tokMinus {
				t.coeff = -t.coeff
			}
			terms = append(terms, t)
		}
		return equation{output: out.text, terms: terms}, nil
	}

	var eqs []equation
	eq, err := parseEq()
	if err != nil {
		return nil, err
	}
	eqs = append(eqs, eq)

	for peek().kind == tokSemicolon {
		advance()
		if peek().kind == tokEOF {
			break
		}
		eq, err := parseEq()
		if err != nil {
			return nil, err
		}
		eqs = append(eqs, eq)
	}

	if peek().kind != tokEOF {
		return nil, fmt.Errorf("sumblk: unexpected token %q: %w", peek().text, ErrInvalidExpression)
	}
	return eqs, nil
}

func SumBlk(expr string, widths ...int) (*System, error) {
	trimmed := strings.TrimSpace(expr)
	if trimmed == "" {
		return nil, fmt.Errorf("sumblk: empty expression: %w", ErrInvalidExpression)
	}

	tokens, err := tokenize(trimmed)
	if err != nil {
		return nil, err
	}

	eqs, err := parseEquations(tokens)
	if err != nil {
		return nil, err
	}

	outputSet := make(map[string]bool)
	var outputs []string
	for _, eq := range eqs {
		if outputSet[eq.output] {
			return nil, fmt.Errorf("sumblk: duplicate output %q: %w", eq.output, ErrInvalidExpression)
		}
		outputSet[eq.output] = true
		outputs = append(outputs, eq.output)
	}

	inputSet := make(map[string]bool)
	var inputs []string
	for _, eq := range eqs {
		for _, t := range eq.terms {
			if !inputSet[t.name] {
				inputSet[t.name] = true
				inputs = append(inputs, t.name)
			}
		}
	}

	signals := make([]string, 0, len(outputs)+len(inputs))
	signals = append(signals, outputs...)
	for _, inp := range inputs {
		if !outputSet[inp] {
			signals = append(signals, inp)
		}
	}

	sigWidth := make(map[string]int, len(signals))
	switch len(widths) {
	case 0:
		for _, s := range signals {
			sigWidth[s] = 1
		}
		for _, inp := range inputs {
			if outputSet[inp] && sigWidth[inp] == 0 {
				sigWidth[inp] = 1
			}
		}
	case 1:
		for _, s := range signals {
			sigWidth[s] = widths[0]
		}
		for _, inp := range inputs {
			if outputSet[inp] {
				sigWidth[inp] = widths[0]
			}
		}
	default:
		if len(widths) != len(signals) {
			return nil, fmt.Errorf("sumblk: got %d widths for %d signals: %w", len(widths), len(signals), ErrInvalidExpression)
		}
		for i, s := range signals {
			sigWidth[s] = widths[i]
		}
	}

	totalRows := 0
	for _, o := range outputs {
		totalRows += sigWidth[o]
	}

	allInputs := make([]string, 0, len(inputs))
	for _, inp := range inputs {
		allInputs = append(allInputs, inp)
	}

	totalCols := 0
	inputColStart := make(map[string]int, len(allInputs))
	for _, inp := range allInputs {
		inputColStart[inp] = totalCols
		totalCols += sigWidth[inp]
	}

	D := mat.NewDense(totalRows, totalCols, nil)

	rowOff := 0
	for _, eq := range eqs {
		w := sigWidth[eq.output]
		merged := make(map[string]float64)
		for _, t := range eq.terms {
			merged[t.name] += t.coeff
		}
		for name, coeff := range merged {
			col0 := inputColStart[name]
			wIn := sigWidth[name]
			diag := w
			if wIn < diag {
				diag = wIn
			}
			for k := 0; k < diag; k++ {
				D.Set(rowOff+k, col0+k, coeff)
			}
		}
		rowOff += w
	}

	sys, err := NewGain(D, 0)
	if err != nil {
		return nil, err
	}

	var expandedOutputs []string
	for _, o := range outputs {
		w := sigWidth[o]
		if w == 1 {
			expandedOutputs = append(expandedOutputs, o)
		} else {
			for k := 1; k <= w; k++ {
				expandedOutputs = append(expandedOutputs, fmt.Sprintf("%s(%d)", o, k))
			}
		}
	}
	var expandedInputs []string
	for _, inp := range allInputs {
		w := sigWidth[inp]
		if w == 1 {
			expandedInputs = append(expandedInputs, inp)
		} else {
			for k := 1; k <= w; k++ {
				expandedInputs = append(expandedInputs, fmt.Sprintf("%s(%d)", inp, k))
			}
		}
	}
	sys.OutputName = expandedOutputs
	sys.InputName = expandedInputs

	return sys, nil
}
