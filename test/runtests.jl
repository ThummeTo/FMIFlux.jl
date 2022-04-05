#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMIFlux
using Test
using FMIZoo

using FMIFlux.FMIImport: fmi2StringToValueReference, fmi2ValueReference

exportingToolsWindows = [("Dymola", "2022x")]
exportingToolsLinux = [("Dymola", "2022x")]

function runtests(exportingTool)
    ENV["EXPORTINGTOOL"] = exportingTool[1]
    ENV["EXPORTINGVERSION"] = exportingTool[2]

    @testset "Testing FMUs exported from $(ENV["EXPORTINGTOOL"]) ($(ENV["EXPORTINGVERSION"]))" begin
        
        @testset "Sensitivities" begin
            include("sens.jl")
        end
        @testset "ME-NeuralFMU (Continuous)" begin
            include("hybrid_ME.jl")
        end
        @testset "ME-NeuralFMU (Discontinuous)" begin
            include("hybrid_ME_dis.jl")
        end
        @testset "CS-NeuralFMU" begin
            include("hybrid_CS.jl")
        end
        @testset "Multiple FMUs" begin
            include("multi.jl")
        end
        @testset "Training modes" begin
            include("train_modes.jl")
        end
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