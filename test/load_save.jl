#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Flux
using DifferentialEquations: Tsit5

import Random
Random.seed!(1234);

t_start = 0.0
t_step = 0.01
t_stop = 5.0
tData = t_start:t_step:t_stop

solver = Tsit5()
fmu = loadFMU("SpringPendulum1D", EXPORTINGTOOL, EXPORTINGVERSION; type = :ME)
numx = length(fmu.modelDescription.stateValueReferences)

solutions = Vector{FMUSolution}(undef, 2)
ps = Vector{Vector{Float64}}(undef, 2)
path = joinpath(@__DIR__, "test.jld2")

for i in 1:2
    net = Chain(x -> fmu(x = x, dx_refs = :all),
                Dense(numx, 16, tanh),
                Dense(16, numx, identity)) 

    problem = ME_NeuralFMU(fmu, net, (t_start, t_stop), solver)
    problem.modifiedState = false
    ps[i] = copy(problem.p)
    @test problem != nothing

    solutions[i] = problem(X0; saveat = tData)

    if i == 1 # save 
        FMIFlux.saveParameters(problem, path)
    elseif i == 2 # load 

        # first, solutions must differ!
        @test !isapprox(solutions[1].states.u[end], solutions[2].states.u[end]; atol=1e-6)
        @test ps[1] != ps[2]

        # load params and re-run
        FMIFlux.loadParameters(problem, path)
        solutions[i] = problem(X0; saveat = tData)
        ps[i] = copy(problem.p)

        # now, solutions must be the same!
        @test isapprox(solutions[1].states.u[end], solutions[2].states.u[end]; atol=1e-6)
        @test ps[1] == ps[2]
    end
end

unloadFMU(fmu)
