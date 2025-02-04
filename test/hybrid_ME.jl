#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Flux
using DifferentialEquations

import Random
Random.seed!(1234);

t_start = 0.0
t_step = 0.01
t_stop = 5.0
tData = t_start:t_step:t_stop

# generate training data
posData, velData, accData = syntTrainingData(tData)

# load FMU for NeuralFMU
fmu = loadFMU("SpringPendulum1D", EXPORTINGTOOL, EXPORTINGVERSION; type = :ME)

# loss function for training
losssum = function (p)
    global problem, X0, posData
    solution = problem(X0; p = p, saveat = tData)

    if !solution.success
        return Inf
    end

    posNet = getState(solution, 1; isIndex = true)
    velNet = getState(solution, 2; isIndex = true)

    return Flux.Losses.mse(posNet, posData) + Flux.Losses.mse(velNet, velData)
end

numStates = length(fmu.modelDescription.stateValueReferences)

# some NeuralFMU setups
nets = []

c1 = CacheLayer()
c2 = CacheRetrieveLayer(c1)
c3 = CacheLayer()
c4 = CacheRetrieveLayer(c3)

init = Flux.glorot_uniform
getVRs = [stringToValueReference(fmu, "mass.s")]
numGetVRs = length(getVRs)
y = zeros(fmi2Real, numGetVRs)
setVRs = [stringToValueReference(fmu, "mass.m")]
numSetVRs = length(setVRs)
setVal = [1.1]

# 1. default ME-NeuralFMU (learn dynamics and states, almost-neutral setup, parameter count << 100)
net = Chain(
    x -> c1(x),
    Dense(numStates, 1, tanh; init = init),
    x -> c2(x[1], 1),
    x -> fmu(; x = x, dx_refs = :all),
    x -> c3(x),
    Dense(numStates, 1, tanh; init = init),
    x -> c4(1, x[1]),
)
push!(nets, net)

# 2. default ME-NeuralFMU (learn dynamics)
net = Chain(
    x -> fmu(; x = x, dx_refs = :all),
    x -> c3(x),
    Dense(numStates, 8, tanh; init = init),
    Dense(8, 16, tanh; init = init),
    Dense(16, 1, tanh; init = init),
    x -> c4(1, x[1]),
)
push!(nets, net)

# 3. default ME-NeuralFMU (learn states)
net = Chain(
    x -> c1(x),
    Dense(numStates, 16, tanh; init = init),
    Dense(16, 1, tanh; init = init),
    x -> c2(x[1], 1),
    x -> fmu(; x = x, dx_refs = :all),
)
push!(nets, net)

# 4. default ME-NeuralFMU (learn dynamics and states)
net = Chain(
    x -> c1(x),
    Dense(numStates, 8, tanh; init = init),
    Dense(8, 16, tanh; init = init),
    Dense(16, 1, tanh; init = init),
    x -> c2(x[1], 1),
    x -> fmu(; x = x, dx_refs = :all),
    x -> c3(x),
    Dense(numStates, 8, tanh, init = init),
    Dense(8, 16, tanh; init = init),
    Dense(16, 1, tanh, init = init),
    x -> c4(1, x[1]),
)
push!(nets, net)

# 5. NeuralFMU with hard setting time to 0.0
net = Chain(
    states -> fmu(; x = states, t = 0.0, dx_refs = :all),
    x -> c3(x),
    Dense(numStates, 8, tanh; init = init),
    Dense(8, 16, tanh; init = init),
    Dense(16, 1, tanh; init = init),
    x -> c4(1, x[1]),
)
push!(nets, net)

# 6. NeuralFMU with additional getter 
net = Chain(
    x -> fmu(; x = x, y_refs = getVRs, dx_refs = :all),
    x -> c3(x),
    Dense(numStates + numGetVRs, 8, tanh; init = init),
    Dense(8, 16, tanh; init = init),
    Dense(16, 1, tanh; init = init),
    x -> c4(1, x[1]),
)
push!(nets, net)

# 7. NeuralFMU with additional setter 
net = Chain(
    x -> fmu(; x = x, u_refs = setVRs, u = setVal, dx_refs = :all),
    x -> c3(x),
    Dense(numStates, 8, tanh; init = init),
    Dense(8, 16, tanh; init = init),
    Dense(16, 1, tanh; init = init),
    x -> c4(1, x[1]),
)
push!(nets, net)

# 8. NeuralFMU with additional setter and getter
net = Chain(
    x -> fmu(; x = x, u_refs = setVRs, u = setVal, y_refs = getVRs, dx_refs = :all),
    x -> c3(x),
    Dense(numStates + numGetVRs, 8, tanh; init = init),
    Dense(8, 16, tanh; init = init),
    Dense(16, 1, tanh; init = init),
    x -> c4(1, x[1]),
)
push!(nets, net)

# 9. an empty NeuralFMU (this does only make sense for debugging)
net = Chain(x -> fmu(x = x, dx_refs = :all))
push!(nets, net)

solvers = [Tsit5()]#, Rosenbrock23(autodiff=false)]

for solver in solvers
    @testset "Solver: $(solver)" begin
        for i = 1:1:length(nets)
            @testset "Net setup $(i)/$(length(nets)) (Continuous NeuralFMU)" begin
                global nets, problem, iterCB
                global LAST_LOSS, FAILED_GRADIENTS

                # if i ∈ (1, 3, 4)
                #     @warn "Currently skipping nets $(i) ∈ (1, 3, 4)"
                #     continue
                # end

                optim = OPTIMISER(ETA)

                net = nets[i]
                problem = ME_NeuralFMU(fmu, net, (t_start, t_stop), solver)
                @test problem != nothing

                # [Note] this is not needed from a mathematical perspective, because the system is continuous differentiable
                if i ∈ (1, 3, 4)
                    problem.modifiedState = true
                end

                # train it ...
                p_net = FMIFlux.params(problem)
                #@test length(p_net) == 1

                solutionBefore = problem(X0; p = p_net, saveat = tData) #  p_net
                if solutionBefore.success
                    @test length(solutionBefore.states.t) == length(tData)
                    @test solutionBefore.states.t[1] == t_start
                    @test solutionBefore.states.t[end] == t_stop
                end

                LAST_LOSS = losssum(p_net)
                @info "Start-Loss for net #$i: $(LAST_LOSS)"

                if length(p_net) == 0
                    @info "The following warning is not an issue, because training on zero parameters must throw a warning:"
                end

                FAILED_GRADIENTS = 0
                FMIFlux.train!(
                    losssum,
                    problem,
                    Iterators.repeated((), NUMSTEPS),
                    optim;
                    gradient = GRADIENT,
                    cb = () -> callback(p_net),
                )
                @info "Failed Gradients: $(FAILED_GRADIENTS) / $(NUMSTEPS)"
                @test FAILED_GRADIENTS <= FAILED_GRADIENTS_QUOTA * NUMSTEPS

                # check results
                solutionAfter = problem(X0; p = p_net, saveat = tData)
                if solutionAfter.success
                    @test length(solutionAfter.states.t) == length(tData)
                    @test solutionAfter.states.t[1] == t_start
                    @test solutionAfter.states.t[end] == t_stop
                end

            end
        end
    end
end

@test length(fmu.components) <= 1

unloadFMU(fmu)
