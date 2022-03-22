#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module FMIFlux

@debug "Debugging messages enabled for FMIFlux ..."

using Requires

using FMIImport
using FMIImport: fmi2ValueReference, FMU, FMU2, FMU2Component

# ToDo: this can not be imported correctly from FMIImport.jl
fmi2Struct = Union{FMU2, FMU2Component}

using FMIImport: fmi2SetupExperiment, fmi2EnterInitializationMode, fmi2ExitInitializationMode, fmi2Reset, fmi2Terminate
using FMIImport: fmi2NewDiscreteStates, fmi2SetContinuousStates, fmi2GetContinuousStates, fmi2GetNominalsOfContinuousStates
using FMIImport: fmi2SetTime, fmi2CompletedIntegratorStep, fmi2GetEventIndicators, fmi2GetDerivatives, fmi2GetReal
using FMIImport: fmi2SampleDirectionalDerivative, fmi2GetDirectionalDerivative, fmi2GetJacobian, fmi2GetJacobian!
using FMIImport: fmi2True, fmi2False

include("FMI2_neural.jl")
include("FMI_neural.jl")
include("misc.jl")

function __init__()
    @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin 
        import .Plots

        include("FMI_plot.jl")

        Plots.plot(nfmu::NeuralFMU) = fmiPlot(nfmu)
    end

    @require FMI="14a09403-18e3-468f-ad8a-74f8dda2d9ac" begin 
        @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin 
            import .FMI 
            
            FMI.fmiPlot(nfmu::NeuralFMU) = fmiPlot(nfmu)
        end
    end
end

# FMI2_neural.jl
export fmi2EvaluateME, fmi2DoStepCS
export fmi2InputDoStepCSOutput
export ME_NeuralFMU, CS_NeuralFMU, NeuralFMU, NeuralFMUInputLayer, NeuralFMUOutputLayer

# FMI_neural.jl
export fmiEvaluateME, fmiDoStepCS
export fmiInputDoStepCSOutput

# misc.jl
export mse_interpolate, transferParams!, lin_interp

# debugging only 
# export _build_jac_dx_x_slow

end # module
