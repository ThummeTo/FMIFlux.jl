#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMIFlux
using Test
using FMIZoo
using FMIFlux.FMIImport
using FMIFlux.FMIImport.FMICore
using FMIFlux.Flux

import FMIFlux.FMISensitivity: FiniteDiff, ForwardDiff, ReverseDiff

using FMIFlux.FMIImport:
    stringToValueReference, fmi2ValueReference, prepareSolveFMU, fmi2Real
using FMIFlux.FMIImport: FMU_EXECUTION_CONFIGURATIONS
using FMIFlux.FMIImport: getState, getValue, getTime

exportingToolsWindows = [("Dymola", "2022x")] # [("ModelicaReferenceFMUs", "0.0.25")]
exportingToolsLinux = [("Dymola", "2022x")]

# number of training steps to perform
global NUMSTEPS = 30
global ETA = 1e-5
global GRADIENT = nothing
global EXPORTINGTOOL = nothing
global EXPORTINGVERSION = nothing
global X0 = [2.0, 0.0]
global OPTIMISER = Descent
global FAILED_GRADIENTS_QUOTA = 1 / 3

# callback for bad optimization steps counter
global FAILED_GRADIENTS = 0
global LAST_LOSS
callback = function (p)
    global LAST_LOSS, FAILED_GRADIENTS
    loss = losssum(p[1])
    if loss >= LAST_LOSS
        FAILED_GRADIENTS += 1
    end
    #@info "$(loss)"
    LAST_LOSS = loss
end

# training data for pendulum experiment 
function syntTrainingData(tData)
    posData = cos.(tData * 3.0) * 2.0
    velData = sin.(tData * 3.0) * -6.0
    accData = cos.(tData * 3.0) * -18.0
    return posData, velData, accData
end

# enable assertions for warnings/errors for all default execution configurations - we want strict tests here
for exec in FMU_EXECUTION_CONFIGURATIONS
    exec.assertOnError = true
    exec.assertOnWarning = true
end

function runtests(exportingTool)

    global EXPORTINGTOOL = exportingTool[1]
    global EXPORTINGVERSION = exportingTool[2]
    @info "Testing FMUs exported from $(EXPORTINGTOOL) ($(EXPORTINGVERSION))"
    @testset "Testing FMUs exported from $(EXPORTINGTOOL) ($(EXPORTINGVERSION))" begin

        @info "Solution Gradients (solution_gradients.jl)"
        @testset "Solution Gradients" begin
            include("solution_gradients.jl")
        end

        @info "Time Event Solution Gradients (time_solution_gradients.jl)"
        @testset "Time Event Solution Gradients" begin
            include("time_solution_gradients.jl")
        end

        for _GRADIENT ∈ (:ReverseDiff, :ForwardDiff) # , :FiniteDiff)

            global GRADIENT = _GRADIENT
            @info "Gradient: $(GRADIENT)"
            @testset "Gradient: $(GRADIENT)" begin

                @info "Layers (layers.jl)"
                @testset "Layers" begin
                    include("layers.jl")
                end

                @info "ME-NeuralFMU (Continuous) (hybrid_ME.jl)"
                @testset "ME-NeuralFMU (Continuous)" begin
                    include("hybrid_ME.jl")
                end

                @info "ME-NeuralFMU (Discontinuous) (hybrid_ME_dis.jl)"
                @testset "ME-NeuralFMU (Discontinuous)" begin
                    include("hybrid_ME_dis.jl")
                end

                @info "NeuralFMU with FMU parameter optimization (fmu_params.jl)"
                @testset "NeuralFMU with FMU parameter optimization" begin
                    include("fmu_params.jl")
                end

                @info "Training modes (train_modes.jl)"
                @testset "Training modes" begin
                    include("train_modes.jl")
                end

                @warn "Multi-threading Test Skipped"
                # @info    "Multi-threading (multi_threading.jl)"
                # @testset "Multi-threading" begin
                #     include("multi_threading.jl")
                # end

                @info "CS-NeuralFMU (hybrid_CS.jl)"
                @testset "CS-NeuralFMU" begin
                    include("hybrid_CS.jl")
                end

                @info "Multiple FMUs (multi.jl)"
                @testset "Multiple FMUs" begin
                    include("multi.jl")
                end

                @info "Batching (batching.jl)"
                @testset "Batching" begin
                    include("batching.jl")
                end

                @info "Optimizers from Optim.jl (optim.jl)"
                @testset "Optim" begin
                    include("optim.jl")
                end

            end
        end

        @info "Checking supported sensitivities skipped"
        # @info    "Benchmark: Supported sensitivities (supported_sensitivities.jl)"
        # @testset "Benchmark: Supported sensitivities " begin
        #     include("supported_sensitivities.jl")
        # end

    end
end

@testset "FMIFlux.jl" begin
    if Sys.iswindows()
        @info "Automated testing is supported on Windows."
        for exportingTool in exportingToolsWindows
            runtests(exportingTool)
        end
    elseif Sys.islinux()
        @info "Automated testing is supported on Linux."
        for exportingTool in exportingToolsLinux
            runtests(exportingTool)
        end
    elseif Sys.isapple()
        @warn "Test-sets are currrently using Windows- and Linux-FMUs, automated testing for macOS is currently not supported."
    end
end
