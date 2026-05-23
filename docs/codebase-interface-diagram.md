# Controlsys Codebase Interface Diagram

This diagram shows the current module interfaces after the PR149-151 architecture deepening work. It is a codebase-level view, not a complete call graph: the public model interfaces are centered, and the internal seams show where recurring rules are localized.

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
        ModelArray["ModelArray<br/>compatible model grid"]
        Generalized["GeneralizedModel / GeneralizedClosedLoop<br/>analysis-point model interface"]
        Tunable["TunableBlock<br/>tunable gain, PID, TF, or SS block"]
        FreqResp["FreqResponseMatrix<br/>sampled complex response"]
        TimeResp["TimeResponse<br/>sampled time-domain output"]
    end

    subgraph construction["Construction and identification"]
        New["New / NewGain / NewFromSlices"]
        NewDescriptor["NewDescriptor / ToExplicit"]
        NewTF["TransferFunc.StateSpace"]
        NewZPK["NewZPK / NewZPKMIMO"]
        NewFRD["NewFRD"]
        NewArray["NewModelArray / StackModelArrays"]
        ERA["ERA<br/>Markov parameters to state-space model"]
        FreqEst["FreqRespEst<br/>sampled input/output to response estimate"]
        Linearize["Linearize / EKF<br/>local nonlinear approximation"]
        Physical["AssemblePhysical<br/>port-checked component assembly"]
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
        StateUtils["StateTransform / EliminateStates<br/>FixedInputReduction"]
    end

    subgraph analysis["Analysis interfaces"]
        TimeAnalysis["Step / Impulse / Initial / Lsim / Simulate / StepInfo"]
        FreqAnalysis["FreqResponse / Bode / Nyquist / Margin / Sigma / FRD helpers"]
        ModelAnalysis["Poles / Zeros / Damp / IsStable / Pzmap"]
        EnergyAnalysis["Gram / HSV / H2Norm / HinfNorm / Covar / Passive"]
        StructureAnalysis["Ctrb / Obsv / Stabilizable / Detectable"]
        LoopAnalysis["Loopsens / RootLocus"]
        Passivity["Passive / FRDPassive / SpectralFactor"]
    end

    subgraph transforms["Transformation and reduction"]
        Realization["SS2SS / Xperm / Canon"]
        Balancing["Balreal / Balred / Modred / Sminreal / ModalTruncate"]
        Decomposition["Stabsep / Modsep / Prescale / Ssbal"]
        Algebra["Inv / Augstate"]
    end

    subgraph synthesis["Design and synthesis"]
        Riccati["Care / Dare / Lyap / DLyap"]
        Controller["Lqr / Dlqr / Lqi / Lqrd / Place / Acker"]
        Observer["Kalman / LQG / Observer assembly"]
        Robust["H2Syn / HinfSyn"]
        PID["Pidtune / PID / SmithPredictor"]
        FixedTuning["Systune / Looptune<br/>tuning goals"]
    end

    caller --> New
    caller --> NewDescriptor
    caller --> NewTF
    caller --> NewZPK
    caller --> NewFRD
    caller --> NewArray
    caller --> ERA
    caller --> FreqEst
    caller --> Linearize
    caller --> Physical
    caller --> Tunable

    New --> System
    NewDescriptor --> System
    NewTF --> System
    NewZPK --> ZPK
    NewFRD --> FRD
    NewArray --> ModelArray
    ERA --> System
    FreqEst --> FreqResp
    FreqEst --> FRD
    Linearize --> System
    Physical --> System
    Tunable --> Generalized

    TF <--> ZPK
    TF --> System
    ZPK --> TF
    ZPK --> System
    System --> TFConv --> TF
    System --> ZPKConv --> ZPK
    System --> FRDConv --> FRD
    FRD --> FreqResp
    ModelArray --> System
    ModelArray --> FreqResp
    ModelArray --> TimeResp
    Generalized --> System

    System --> Series --> System
    System --> Parallel --> System
    System --> Feedback --> System
    System --> AppendConnect --> System
    System --> DelayOps --> System

    System --> C2D --> System
    System --> D2C --> System
    System --> StateUtils --> System

    System --> TimeAnalysis --> TimeResp
    System --> FreqAnalysis --> FreqResp
    FRD --> FreqAnalysis
    System --> ModelAnalysis
    System --> EnergyAnalysis
    System --> StructureAnalysis
    System --> LoopAnalysis
    System --> Passivity
    FRD --> Passivity

    System --> Realization --> System
    System --> Balancing --> System
    System --> Decomposition --> System
    System --> Algebra --> System

    System --> Controller
    System --> Observer
    System --> Robust
    System --> PID
    System --> Generalized
    Generalized --> FixedTuning
    Tunable --> FixedTuning
    Riccati --> Controller
    Riccati --> Observer
    Riccati --> Robust
    Controller --> System
    Observer --> System
    Robust --> System
    PID --> System
    FixedTuning --> System
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
        +Abs()
        +SelectFrequencies()
        +MapResponse()
        +PeakGain()
        +FreqResponse()
        +Bode()
    }

    class ModelArray {
        +Shape()
        +Model()
        +SelectFlat()
        +FreqResponse()
        +Step()
    }

    class GeneralizedModel {
        +InsertAnalysisPoint()
        +AnalysisPoint()
        +CurrentSystem()
    }

    class GeneralizedClosedLoop {
        +OpenLoop()
        +ClosedLoop()
        +Sensitivity()
        +ComplementarySensitivity()
    }

    class TunableBlock {
        +CurrentSystem()
        +FreeParameters()
        +SampleBlock()
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

    class stateSpaceUtilitySeam {
        +NewDescriptor()
        +ToExplicit()
        +EliminateStates()
        +FixedInputReduction()
        +AugmentInternalDelayOutputs()
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

    class generalizedTuningSeam {
        +analysisPoints()
        +openLoop()
        +closedLoop()
        +tunableController()
    }

    class tuningGoalEvaluator {
        +tracking()
        +maxGain()
        +loopShape()
        +margin()
        +pole()
        +overshoot()
    }

    class passivitySeam {
        +Passive()
        +FRDPassive()
        +SpectralFactor()
    }

    class physicalAssemblySeam {
        +validatePorts()
        +prefixMetadata()
        +appendComponents()
    }

    class modelArraySeam {
        +validateCompatible()
        +flatIndex()
        +freqResponse()
        +step()
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
    System --> stateSpaceUtilitySeam : state-space utilities
    System --> frequencyEvaluator : frequency response
    System --> timeResponsePlanner : time response
    System --> simulationDispatcher : sampled simulation
    System --> passivitySeam : passivity and spectral factor
    System --> physicalAssemblySeam : physical component assembly
    System --> ModelArray : compatible model arrays

    TransferFunc --> System : realization
    ZPK --> TransferFunc : rational-channel conversion
    FRD --> sampledResponseLayout : sampled response access
    frequencyEvaluator --> sampledResponseLayout : flat response layout
    ModelArray --> modelArraySeam : array validation and analysis
    GeneralizedModel --> generalizedTuningSeam : analysis-point model wrapper
    GeneralizedClosedLoop --> generalizedTuningSeam : loop extraction
    TunableBlock --> generalizedTuningSeam : sampled controller blocks

    generalizedPlantPartition --> System : H2/Hinf controller synthesis
    generalizedTuningSeam --> tuningGoalEvaluator : fixed-structure tuning
    tuningGoalEvaluator --> System : evaluates closed-loop model
    controllerObserverPolicy --> System : regulator and estimator assembly
    matrixEquationProblem --> controllerObserverPolicy : Riccati and Lyapunov validation
```

## Interface Reading Guide

- `System` is the fundamental representation. Most public workflows either consume it, return it, or convert another model interface into it.
- `TransferFunc`, `ZPK`, `FRD`, `ModelArray`, and generalized/tunable model wrappers are alternate caller-facing interfaces. They preserve input/output names, sample time, and analysis-point metadata where the representation supports them.
- Interconnection routines concentrate compatibility checks, direct feedthrough handling, delay movement, and metadata propagation behind a small caller-facing interface.
- Delay behavior is intentionally split between topology and conversion seams: topology answers what delay structure exists; conversion decides whether it remains explicit, becomes a delay bank, or moves into LFT form.
- Analysis routines share sampled-response layouts so frequency-response data, Bode results, singular-value analysis, and frequency-response estimates use the same output/input/frequency indexing.
- Model-array, physical-assembly, and state-space utility seams make MATLAB-parity workflows available while keeping compatibility checks and metadata rules localized.
- Synthesis routines route generalized-plant, generalized tuning, and controller/observer rules through policy modules before returning controller or closed-loop state-space models.
