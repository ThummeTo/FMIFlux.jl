#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMIFlux
using Test

@testset "FMIFlux.jl" begin
    if Sys.iswindows()
        @info "Automated testing for Windows is supported."
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
    elseif Sys.islinux()
        @warn "Test-sets are using Windows-FMUs, automated testing for Linux is currently not supported."
    elseif Sys.isapple()
        @warn "Test-sets are using Windows-FMUs, automated testing for macOS is currently not supported."
    end
end
