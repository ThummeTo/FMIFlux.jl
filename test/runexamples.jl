#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Test

examples = ["simple_hybrid_ME", "simple_hybrid_CS", "modelica_conference_2021", "advanced_hybrid_ME"]

@testset "FMIFlux.jl Examples" begin
    for example in examples
        @testset "$(example).jl" begin
            path = joinpath(dirname(@__FILE__), "..", "example", example * ".jl")
            @test include(path) == nothing
        end
    end
end
