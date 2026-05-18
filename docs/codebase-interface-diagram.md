# Controlsys Codebase Interface Diagram

This diagram shows the current module interfaces after the architecture deepening work. It is a codebase-level view, not a complete call graph: the public model interfaces are centered, and the internal seams show where recurring rules are localized.

## Public Interface Map

Rendered SVG: [codebase-public-interface-map.svg](codebase-public-interface-map.svg)

```mermaid
flowchart LR
    caller["External callers"]

    subgraph models["Model interfaces"]
        System["System<br/>fundamental state-space model<br/>A, B, C, D, optional E, delays, names"]
        TF["TransferFunc<br/>polynomial-ratio model"]
        ZPK["ZPK<br/>zero-pole-gain model"]
        FRD["FRD<br/>frequency-response data model"]
        FreqResp["FreqResponseMatrix<br/>sampled complex response"]
        TimeResp["TimeResponse<br/>sampled time-domain output"]
    end

    subgraph construction["Construction and identification"]
        New["New / NewGain / NewFromSlices"]
        NewTF["TransferFunc.StateSpace"]
        NewZPK["NewZPK / NewZPKMIMO"]
        NewFRD["NewFRD"]
        ERA["ERA<br/>Markov parameters to state-space model"]
        FreqEst["FreqRespEst<br/>sampled input/output to response estimate"]
        Linearize["Linearize / EKF<br/>local nonlinear approximation"]
    end

    subgraph interconnection["Interconnection interfaces"]
        Series["Series"]
        Parallel["Parallel"]
        Feedback["Feedback / SafeFeedback"]
        AppendConnect["Append / Connect / LFT / SumBlk"]
        DelayOps["PadeDelay / ThiranDelay<br/>PullDelaysToLFT / AbsorbDelay"]
    end

    subgraph conversion["Representation and domain conversion"]
        TFConv["System.TransferFunction"]
        ZPKConv["System.ZPKModel"]
        FRDConv["System.FRD"]
        C2D["Discretize / DiscretizeWithOpts"]
        D2C["Undiscretize / D2C"]
    end

    subgraph analysis["Analysis interfaces"]
        TimeAnalysis["Step / Impulse / Initial / Lsim / Simulate"]
        FreqAnalysis["FreqResponse / Bode / Nyquist / Margin / Sigma"]
        ModelAnalysis["Poles / Zeros / Damp / IsStable / Pzmap"]
        EnergyAnalysis["Gram / HSV / H2Norm / HinfNorm / Covar"]
        StructureAnalysis["Ctrb / Obsv / Stabilizable / Detectable"]
        LoopAnalysis["Loopsens / RootLocus"]
    end

    subgraph transforms["Transformation and reduction"]
        Realization["SS2SS / Xperm / Canon"]
        Balancing["Balreal / Balred / Modred / Sminreal"]
        Decomposition["Stabsep / Modsep / Prescale / Ssbal"]
        Algebra["Inv / Augstate"]
    end

    subgraph synthesis["Design and synthesis"]
        Riccati["Care / Dare / Lyap / DLyap"]
        Controller["Lqr / Dlqr / Lqi / Lqrd / Place / Acker"]
        Observer["Kalman / LQG / Observer assembly"]
        Robust["H2Syn / HinfSyn"]
        PID["Pidtune / PID / SmithPredictor"]
    end

    caller --> New
    caller --> NewTF
    caller --> NewZPK
    caller --> NewFRD
    caller --> ERA
    caller --> FreqEst
    caller --> Linearize

    New --> System
    NewTF --> System
    NewZPK --> ZPK
    NewFRD --> FRD
    ERA --> System
    FreqEst --> FreqResp
    FreqEst --> FRD
    Linearize --> System

    TF <--> ZPK
    TF --> System
    ZPK --> TF
    ZPK --> System
    System --> TFConv --> TF
    System --> ZPKConv --> ZPK
    System --> FRDConv --> FRD
    FRD --> FreqResp

    System --> Series --> System
    System --> Parallel --> System
    System --> Feedback --> System
    System --> AppendConnect --> System
    System --> DelayOps --> System

    System --> C2D --> System
    System --> D2C --> System

    System --> TimeAnalysis --> TimeResp
    System --> FreqAnalysis --> FreqResp
    FRD --> FreqAnalysis
    System --> ModelAnalysis
    System --> EnergyAnalysis
    System --> StructureAnalysis
    System --> LoopAnalysis

    System --> Realization --> System
    System --> Balancing --> System
    System --> Decomposition --> System
    System --> Algebra --> System

    System --> Controller
    System --> Observer
    System --> Robust
    System --> PID
    Riccati --> Controller
    Riccati --> Observer
    Riccati --> Robust
    Controller --> System
    Observer --> System
    Robust --> System
    PID --> System
```

## Internal Seam Map

Rendered SVG: [codebase-internal-seam-map.svg](codebase-internal-seam-map.svg)

```mermaid
classDiagram
    class System {
        +Dims()
        +Validate()
        +Poles()
        +IsStable()
        +IsContinuous()
        +IsDiscrete()
        +Simulate()
        +FreqResponse()
        +TransferFunction()
    }

    class TransferFunc {
        +Dims()
        +Eval()
        +EvalMulti()
        +StateSpace()
        +ZPK()
    }

    class ZPK {
        +Dims()
        +Eval()
        +FreqResponse()
        +TransferFunction()
        +StateSpace()
    }

    class FRD {
        +Dims()
        +NumFrequencies()
        +At()
        +FreqResponse()
        +Bode()
    }

    class descriptorPolicy {
        +validate()
        +poles()
        +requireStandard()
        +requireRiccatiStandard()
    }

    class timeDomain {
        +validateSampleTime()
        +frequencyVariable()
        +ensureCompatible()
    }

    class delayTopology {
        +totalExternal()
        +decomposedExternal()
        +decomposableExternal()
    }

    class delayConversionPolicy {
        +applyDiscreteDelayFields()
        +applyContinuousDelayFields()
        +replaceDiscreteExternal()
        +replaceContinuousExternal()
    }

    class interconnectionTopology {
        +seriesDelayPlan()
        +parallelDelayPlan()
        +seriesRequiresLFT()
        +parallelRequiresLFT()
    }

    class realizationTransformPolicy {
        +requireStandard()
        +requireDelayFree()
        +result()
        +resultWithOriginalFeedthrough()
        +resultWithZeroFeedthrough()
    }

    class frequencyEvaluator {
        +response()
        +eval()
    }

    class timeResponsePlanner {
        +auto()
        +lsim()
    }

    class simulationDispatcher {
        +run()
    }

    class sampledResponseLayout {
        +offset()
        +blockOffset()
    }

    class generalizedPlantPartition {
        +validateControllerChannels()
        +newController()
        +closedLoopPoles()
    }

    class controllerObserverPolicy {
        +validateNoise()
        +regulator()
        +estimator()
    }

    class matrixEquationProblem {
        +riccatiProblem
        +lyapunovProblem
    }

    System --> descriptorPolicy : descriptor gate
    System --> timeDomain : time-domain rules
    System --> delayTopology : external delay decomposition
    System --> delayConversionPolicy : delay conversion
    System --> interconnectionTopology : interconnection planning
    System --> realizationTransformPolicy : realization assembly
    System --> frequencyEvaluator : frequency response
    System --> timeResponsePlanner : time response
    System --> simulationDispatcher : sampled simulation

    TransferFunc --> System : realization
    ZPK --> TransferFunc : rational-channel conversion
    FRD --> sampledResponseLayout : sampled response access
    frequencyEvaluator --> sampledResponseLayout : flat response layout

    generalizedPlantPartition --> System : H2/Hinf controller synthesis
    controllerObserverPolicy --> System : regulator and estimator assembly
    matrixEquationProblem --> controllerObserverPolicy : Riccati and Lyapunov validation
```

## Interface Reading Guide

- `System` is the fundamental representation. Most public workflows either consume it, return it, or convert another model interface into it.
- `TransferFunc`, `ZPK`, and `FRD` are alternate model interfaces. They preserve input/output names and sample time where the representation supports them.
- Interconnection routines concentrate compatibility checks, direct feedthrough handling, delay movement, and metadata propagation behind a small caller-facing interface.
- Delay behavior is intentionally split between topology and conversion seams: topology answers what delay structure exists; conversion decides whether it remains explicit, becomes a delay bank, or moves into LFT form.
- Analysis routines share sampled-response layouts so frequency-response data, Bode results, singular-value analysis, and frequency-response estimates use the same output/input/frequency indexing.
- Synthesis routines route generalized-plant and controller/observer rules through policy modules before returning controller state-space models.
