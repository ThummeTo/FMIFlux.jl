#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Flux
using DifferentialEquations: Tsit5, Rosenbrock23

import Random
Random.seed!(5678);

t_start = 0.0
t_step = 0.01
t_stop = 5.0
tData = t_start:t_step:t_stop

# generate training data
posData, velData, accData = syntTrainingData(tData)

# load FMU for training
fmu = loadFMU("SpringFrictionPendulum1D", EXPORTINGTOOL, EXPORTINGVERSION; type = :ME)

# loss function for training
losssum = function (p)
    global problem, X0, posData
    solution = problem(X0; p = p, showProgress = true, saveat = tData)

    if !solution.success
        return Inf
    end

    # posNet = getState(solution, 1; isIndex=true)
    velNet = getState(solution, 2; isIndex = true)

    return FMIFlux.Losses.mse(velNet, velData) # Flux.Losses.mse(posNet, posData)
end

# callback function for training
global iterCB = 0
global lastLoss = 0.0
callb = function (p)
    global iterCB += 1
    global lastLoss

    if iterCB % 5 == 0
        loss = losssum(p[1])
        @info "[$(iterCB)] Loss: $loss"

        # This test condition is not good, because when the FMU passes an event, the error might increase.  
        @test (loss < lastLoss) && (loss != lastLoss)
        lastLoss = loss
    end
end

numStates = length(fmu.modelDescription.stateValueReferences)

# some NeuralFMU setups
nets = []

c1 = CacheLayer()
c2 = CacheRetrieveLayer(c1)
c3 = CacheLayer()
c4 = CacheRetrieveLayer(c3)

# 1. Discontinuous ME-NeuralFMU (learn dynamics and states)
net = Chain(
    x -> c1(x),
    Dense(numStates, 16, tanh),
    Dense(16, 1, identity),
    x -> c2(x[1], 1),
    x -> fmu(; x = x, dx_refs = :all),
    x -> c3(x),
    Dense(numStates, 16, tanh),
    Dense(16, 16, tanh),
    Dense(16, 1, identity),
    x -> c4(1, x[1]),
)
push!(nets, net)

for i = 1:length(nets)
    @testset "Net setup $(i)/$(length(nets))" begin
        global nets, problem, lastLoss, iterCB

        net = nets[i]
        solver = Tsit5()
        problem = ME_NeuralFMU(fmu, net, (t_start, t_stop), solver; saveat = tData)

        @test problem !== nothing

        solutionBefore = problem(X0)
        if solutionBefore.success
            @test length(solutionBefore.states.t) == length(tData)
            @test solutionBefore.states.t[1] == t_start
            @test solutionBefore.states.t[end] == t_stop
        end

        # train it ...
        p_net = Flux.params(problem)
        p_start = copy(p_net[1])

        iterCB = 0
        lastLoss = losssum(p_net[1])
        startLoss = lastLoss
        @info "[ $(iterCB)] Loss: $lastLoss"

        p_net[1][:] = p_start[:]
        lastLoss = startLoss
        st = time()
        optim = OPTIMISER(ETA)
        FMIFlux.train!(
            losssum,
            problem,
            Iterators.repeated((), NUMSTEPS),
            optim;
            cb = () -> callb(p_net),
            multiThreading = false,
            gradient = GRADIENT,
        )
        dt = round(time() - st; digits = 2)
        @info "Training time single threaded (not pre-compiled): $(dt)s"

        p_net[1][:] = p_start[:]
        lastLoss = startLoss
        st = time()
        optim = OPTIMISER(ETA)
        FMIFlux.train!(
            losssum,
            problem,
            Iterators.repeated((), NUMSTEPS),
            optim;
            cb = () -> callb(p_net),
            multiThreading = false,
            gradient = GRADIENT,
        )
        dt = round(time() - st; digits = 2)
        @info "Training time single threaded (pre-compiled): $(dt)s"

        # [ToDo] currently not implemented 

        # p_net[1][:] = p_start[:]
        # lastLoss = startLoss
        # st = time()
        # optim = OPTIMISER(ETA)
        # FMIFlux.train!(losssum, problem, Iterators.repeated((), NUMSTEPS), optim; cb=()->callb(p_net), multiThreading=true, gradient=GRADIENT)
        # dt = round(time()-st; digits=2)
        # @info "Training time multi threaded x$(Threads.nthreads()) (not pre-compiled): $(dt)s"

        # p_net[1][:] = p_start[:]
        # lastLoss = startLoss
        # st = time()
        # optim = OPTIMISER(ETA)
        # FMIFlux.train!(losssum, problem, Iterators.repeated((), NUMSTEPS), optim; cb=()->callb(p_net), multiThreading=true, gradient=GRADIENT)
        # dt = round(time()-st; digits=2)
        # @info "Training time multi threaded x$(Threads.nthreads()) (pre-compiled): $(dt)s"

        # check results
        solutionAfter = problem(X0)
        if solutionAfter.success
            @test length(solutionAfter.states.t) == length(tData)
            @test solutionAfter.states.t[1] == t_start
            @test solutionAfter.states.t[end] == t_stop
        end
    end
end

unloadFMU(fmu)
