#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module FMIFlux

@debug "Debugging messages enabled for FMIFlux ..."

using Requires

using FMIImport
using FMIImport: fmi2ValueReference, FMU, FMU2, FMU2Component

#import FMIImport.PreallocationTools

# ToDo: this can not be imported correctly from FMIImport.jl
fmi2Struct = Union{FMU2, FMU2Component}

using FMIImport: fmi2SetupExperiment, fmi2EnterInitializationMode, fmi2ExitInitializationMode, fmi2Reset, fmi2Terminate
using FMIImport: fmi2NewDiscreteStates, fmi2SetContinuousStates, fmi2GetContinuousStates, fmi2GetNominalsOfContinuousStates
using FMIImport: fmi2SetTime, fmi2CompletedIntegratorStep, fmi2GetEventIndicators, fmi2GetDerivatives, fmi2GetReal
using FMIImport: fmi2SampleDirectionalDerivative, fmi2GetDirectionalDerivative, fmi2GetJacobian, fmi2GetJacobian!
using FMIImport: fmi2True, fmi2False

include("FMI_neural.jl")
include("misc.jl")
include("layers.jl")
include("deprecated.jl")
include("losses.jl")

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

# FMI_neural.jl
export ME_NeuralFMU, CS_NeuralFMU, NeuralFMU

# misc.jl
export mse_interpolate, transferParams!, transferFlatParams!, lin_interp

# deprecated.jl
# >>> deprecated functions are exported inside the file itself

end # module
