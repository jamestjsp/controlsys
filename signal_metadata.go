package controlsys

type signalMetadata struct {
	input  []string
	output []string
	state  []string
	notes  string
}

func metadataFromSystem(sys *System) signalMetadata {
	return signalMetadata{
		input:  sys.InputName,
		output: sys.OutputName,
		state:  sys.StateName,
		notes:  sys.Notes,
	}
}

func (md signalMetadata) applyAll(sys *System) {
	sys.InputName = copyStringSlice(md.input)
	sys.OutputName = copyStringSlice(md.output)
	sys.StateName = copyStringSlice(md.state)
	sys.Notes = md.notes
}

func (md signalMetadata) applyIO(sys *System) {
	sys.InputName = copyStringSlice(md.input)
	sys.OutputName = copyStringSlice(md.output)
}

func (md signalMetadata) applyAllOwned(sys *System) {
	sys.InputName = md.input
	sys.OutputName = md.output
	sys.StateName = md.state
	sys.Notes = md.notes
}

func (md signalMetadata) applyIOOwned(sys *System) {
	sys.InputName = md.input
	sys.OutputName = md.output
}

func (md signalMetadata) selectIO(inputs, outputs []int) signalMetadata {
	return signalMetadata{
		input:  selectStringSlice(md.input, inputs),
		output: selectStringSlice(md.output, outputs),
		state:  copyStringSlice(md.state),
		notes:  md.notes,
	}
}

func seriesMetadata(left, right *System, n1, n2 int) signalMetadata {
	return signalMetadata{
		input:  copyStringSlice(left.InputName),
		output: copyStringSlice(right.OutputName),
		state:  concatStringSlices([][]string{left.StateName, right.StateName}, []int{n1, n2}),
	}
}

func controllerMetadata(measurementNames, controlNames []string) signalMetadata {
	return signalMetadata{
		input:  measurementNames,
		output: controlNames,
	}
}

func fohStateMetadata(src *System, n, m int) []string {
	if src.StateName == nil && src.InputName == nil {
		return nil
	}
	names := make([]string, n+m)
	if src.StateName != nil {
		copy(names, src.StateName)
	}
	for j := 0; j < m; j++ {
		if src.InputName != nil && j < len(src.InputName) {
			names[n+j] = src.InputName[j] + "_prev"
		}
	}
	return names
}
