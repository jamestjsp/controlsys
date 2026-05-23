package controlsys

import "fmt"

type PhysicalPortKind int

const (
	PhysicalPortDisplacement PhysicalPortKind = iota
	PhysicalPortEffort
)

type PhysicalPort struct {
	Name      string
	Kind      PhysicalPortKind
	Dimension int
}

type PhysicalComponent struct {
	Name   string
	System *System
	Ports  []PhysicalPort
}

type PhysicalConnection struct {
	FromComponent string
	FromPort      string
	ToComponent   string
	ToPort        string
	Grounded      bool
}

func NewPhysicalComponent(name string, sys *System, ports []PhysicalPort) PhysicalComponent {
	return PhysicalComponent{Name: name, System: sys.Copy(), Ports: append([]PhysicalPort(nil), ports...)}
}

func AssemblePhysical(name string, components []PhysicalComponent, connections []PhysicalConnection) (*System, error) {
	if len(components) == 0 {
		return nil, fmt.Errorf("AssemblePhysical: no components: %w", ErrDimensionMismatch)
	}
	byName := make(map[string]PhysicalComponent, len(components))
	for _, component := range components {
		if component.Name == "" || component.System == nil {
			return nil, fmt.Errorf("AssemblePhysical: invalid component: %w", ErrDimensionMismatch)
		}
		if _, exists := byName[component.Name]; exists {
			return nil, fmt.Errorf("AssemblePhysical: duplicate component %q: %w", component.Name, ErrDimensionMismatch)
		}
		byName[component.Name] = component
	}
	for _, conn := range connections {
		from, err := lookupPhysicalPort(byName, conn.FromComponent, conn.FromPort)
		if err != nil {
			return nil, err
		}
		if conn.Grounded {
			continue
		}
		to, err := lookupPhysicalPort(byName, conn.ToComponent, conn.ToPort)
		if err != nil {
			return nil, err
		}
		if from.Kind != to.Kind || from.Dimension != to.Dimension {
			return nil, fmt.Errorf("AssemblePhysical: incompatible ports %s.%s and %s.%s: %w",
				conn.FromComponent, conn.FromPort, conn.ToComponent, conn.ToPort, ErrDimensionMismatch)
		}
	}

	systems := make([]*System, len(components))
	for i, component := range components {
		sys := component.System.Copy()
		prefixSystemMetadata(sys, component.Name)
		systems[i] = sys
	}
	if len(systems) == 1 {
		return systems[0], nil
	}
	assembled := systems[0]
	for _, sys := range systems[1:] {
		next, err := Append(assembled, sys)
		if err != nil {
			return nil, err
		}
		assembled = next
	}
	return assembled, nil
}

func lookupPhysicalPort(components map[string]PhysicalComponent, componentName, portName string) (PhysicalPort, error) {
	component, ok := components[componentName]
	if !ok {
		return PhysicalPort{}, fmt.Errorf("AssemblePhysical: component %q not found: %w", componentName, ErrSignalNotFound)
	}
	for _, port := range component.Ports {
		if port.Name == portName {
			if port.Dimension <= 0 {
				return PhysicalPort{}, fmt.Errorf("AssemblePhysical: invalid port dimension: %w", ErrDimensionMismatch)
			}
			return port, nil
		}
	}
	return PhysicalPort{}, fmt.Errorf("AssemblePhysical: port %q not found on %q: %w", portName, componentName, ErrSignalNotFound)
}

func prefixSystemMetadata(sys *System, prefix string) {
	for i, name := range sys.InputName {
		sys.InputName[i] = prefix + "." + name
	}
	for i, name := range sys.OutputName {
		sys.OutputName[i] = prefix + "." + name
	}
	for i, name := range sys.StateName {
		sys.StateName[i] = prefix + "." + name
	}
}
