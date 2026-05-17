package controlsys

import "fmt"

func trimLeadingFloatZeros(p []float64) []float64 {
	i := 0
	for i < len(p) && p[i] == 0 {
		i++
	}
	if i == len(p) {
		return []float64{0}
	}
	return p[i:]
}

func channelPolynomialGain(num, den []float64) float64 {
	num = trimLeadingFloatZeros(num)
	den = trimLeadingFloatZeros(den)
	if len(num) == 1 && num[0] == 0 {
		return 0
	}
	return num[0] / den[0]
}

func validateTransferChannelShape(tf *TransferFunc) (p, m int, err error) {
	if tf == nil {
		return 0, 0, fmt.Errorf("TransferFunc: nil model: %w", ErrDimensionMismatch)
	}
	p = len(tf.Den)
	if p == 0 {
		return 0, 0, nil
	}
	if len(tf.Num) != p {
		return 0, 0, fmt.Errorf("TransferFunc: numerator has %d rows, want %d: %w", len(tf.Num), p, ErrDimensionMismatch)
	}
	m = len(tf.Num[0])
	if m == 0 {
		return 0, 0, fmt.Errorf("TransferFunc: zero input channels: %w", ErrDimensionMismatch)
	}
	if tf.Delay != nil && len(tf.Delay) != p {
		return 0, 0, fmt.Errorf("TransferFunc: delay has %d rows, want %d: %w", len(tf.Delay), p, ErrDimensionMismatch)
	}
	for i := range p {
		if len(tf.Den[i]) == 0 {
			return 0, 0, fmt.Errorf("TransferFunc: row %d has empty denominator: %w", i, ErrDimensionMismatch)
		}
		if len(tf.Num[i]) != m {
			return 0, 0, fmt.Errorf("TransferFunc: row %d has %d numerator channels, want %d: %w", i, len(tf.Num[i]), m, ErrDimensionMismatch)
		}
		if tf.Delay != nil && len(tf.Delay[i]) != m {
			return 0, 0, fmt.Errorf("TransferFunc: row %d has %d delays, want %d: %w", i, len(tf.Delay[i]), m, ErrDimensionMismatch)
		}
		for j := range m {
			if len(tf.Num[i][j]) == 0 {
				return 0, 0, fmt.Errorf("TransferFunc: channel (%d,%d) has empty numerator: %w", i, j, ErrDimensionMismatch)
			}
		}
	}
	return p, m, nil
}

func validateZPKChannelShape(z *ZPK) (p, m int, err error) {
	if z == nil {
		return 0, 0, fmt.Errorf("ZPK: nil model: %w", ErrDimensionMismatch)
	}
	p = len(z.Gain)
	if p == 0 {
		return 0, 0, fmt.Errorf("ZPK: zero output channels: %w", ErrDimensionMismatch)
	}
	m = len(z.Gain[0])
	if m == 0 {
		return 0, 0, fmt.Errorf("ZPK: zero input channels: %w", ErrDimensionMismatch)
	}
	if len(z.Zeros) != p || len(z.Poles) != p {
		return 0, 0, fmt.Errorf("ZPK: channel rows do not match gain rows: %w", ErrDimensionMismatch)
	}
	for i := range p {
		if len(z.Gain[i]) != m || len(z.Zeros[i]) != m || len(z.Poles[i]) != m {
			return 0, 0, fmt.Errorf("ZPK: row %d channel count mismatch: %w", i, ErrDimensionMismatch)
		}
	}
	return p, m, nil
}
