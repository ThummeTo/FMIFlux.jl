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
posData = collect(abs(cos(u .* 1.0)) for u in tData) * 2.0

fmu = loadFMU("BouncingBall1D", "Dymola", "2023x"; type = :ME) # , logLevel = :info)

# loss function for training
losssum = function (p)
    global problem, X0, posData
    global solution
    solution = problem(X0; p = p, saveat = tData)

    if !solution.success
        return Inf
    end

    posNet = getState(solution, 1; isIndex = true)
    #velNet = getState(solution, 2; isIndex=true)

    return Flux.Losses.mse(posNet, posData) #+ Flux.Losses.mse(velNet, velData) 
end

numStates = length(fmu.modelDescription.stateValueReferences)

# some NeuralFMU setups
nets = []

c1 = CacheLayer()
c2 = CacheRetrieveLayer(c1)
c3 = CacheLayer()
c4 = CacheRetrieveLayer(c3)

init = Flux.glorot_uniform
getVRs = [stringToValueReference(fmu, "mass_s")]
numGetVRs = length(getVRs)
y = zeros(fmi2Real, numGetVRs)
setVRs = [stringToValueReference(fmu, "damping")]
numSetVRs = length(setVRs)
setVal = [0.8]

# 1. default ME-NeuralFMU (learn dynamics and states, almost-neutral setup, parameter count << 100)
net1 = function ()
    net = Chain(
        x -> c1(x),
        Dense(numStates, 1, tanh; init = init),
        x -> c2(x[1], 1),
        x -> fmu(; x = x, dx_refs = :all),
        x -> c3(x),
        Dense(numStates, 1, tanh; init = init),
        x -> c4(1, x[1]),
    )
end
push!(nets, net1)

# 2. default ME-NeuralFMU (learn dynamics)
net2 = function ()
    net = Chain(
        x -> fmu(; x = x, dx_refs = :all),
        x -> c3(x),
        Dense(numStates, 8, tanh; init = init),
        Dense(8, 16, tanh; init = init),
        Dense(16, 1, tanh; init = init),
        x -> c4(1, x[1]),
    )
end
push!(nets, net2)

# 3. default ME-NeuralFMU (learn states)
net3 = function ()
    net = Chain(
        x -> c1(x),
        Dense(numStates, 16, tanh; init = init),
        Dense(16, 1, tanh; init = init),
        x -> c2(x[1], 1),
        x -> fmu(; x = x, dx_refs = :all),
    )
end
push!(nets, net3)

# 4. default ME-NeuralFMU (learn dynamics and states)
net4 = function ()
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
end
push!(nets, net4)

# 5. NeuralFMU with hard setting time to 0.0
net5 = function ()
    net = Chain(
        states -> fmu(; x = states, t = 0.0, dx_refs = :all),
        x -> c3(x),
        Dense(numStates, 8, tanh; init = init),
        Dense(8, 16, tanh; init = init),
        Dense(16, 1, tanh; init = init),
        x -> c4(1, x[1]),
    )
end
push!(nets, net5)

# 6. NeuralFMU with additional getter 
net6 = function ()
    net = Chain(
        x -> fmu(; x = x, y_refs = getVRs, dx_refs = :all),
        dx_y -> c3(dx_y),
        Dense(numStates + numGetVRs, 8, tanh; init = init),
        Dense(8, 16, tanh; init = init),
        Dense(16, 1, tanh; init = init),
        dx -> c4(1, dx[1]),
    )
end
push!(nets, net6)

# 7. NeuralFMU with additional setter 
net7 = function ()
    net = Chain(
        x -> fmu(; x = x, u_refs = setVRs, u = setVal, dx_refs = :all),
        x -> c3(x),
        Dense(numStates, 8, tanh; init = init),
        Dense(8, 16, tanh; init = init),
        Dense(16, 1, tanh; init = init),
        x -> c4(1, x[1]),
    )
end
push!(nets, net7)

# 8. NeuralFMU with additional setter and getter 
net8 = function ()
    net = Chain(
        x -> fmu(; x = x, u_refs = setVRs, u = setVal, y_refs = getVRs, dx_refs = :all),
        x -> c3(x),
        Dense(numStates + numGetVRs, 8, tanh; init = init),
        Dense(8, 16, tanh; init = init),
        Dense(16, 1, tanh; init = init),
        x -> c4(1, x[1]),
    )
end
push!(nets, net8)

# 9. an empty NeuralFMU (this does only make sense for debugging) 
net9 = function ()
    net = Chain(x -> fmu(x = x, dx_refs = :all))
end
push!(nets, net9)

solvers = [Tsit5()]#, Rosenbrock23(autodiff=false)]

for solver in solvers
  @testset "Solver: $(solver)" begin
      for i = 6:6 # 1:length(nets)
          @testset "Net setup $(i)/$(length(nets)) (Discontinuous NeuralFMU)" begin
                global nets, problem, iterCB
                global LAST_LOSS, FAILED_GRADIENTS

                if i ∈ (1, 3, 4)
                    @warn "Currently skipping net $(i) ∈ (1, 3, 4) for disc. FMUs (ANN before FMU)"
                    continue
                end

                optim = OPTIMISER(ETA)

                net_constructor = nets[i]
                problem = nothing
                p_net = nothing

                tries = 0
                maxtries = 1000
                while true
                    net = net_constructor()
                    problem = ME_NeuralFMU(fmu, net, (t_start, t_stop), solver)

                    if i ∈ (1, 3, 4)
                        problem.modifiedState = true
                    end

                    p_net = FMIFlux.params(problem)
                    solutionBefore = problem(X0; p = p_net, saveat = tData)
                    ne = length(solutionBefore.events)

                    if ne > 0 && ne <= 10
                        break
                    else
                        if tries >= maxtries
                            @warn "Solution before did not trigger an acceptable event count (=$(ne) ∉ [1,10]) for net $(i)! Can't find a valid start configuration ($(maxtries) tries)!"
                            break
                        end
                        tries += 1
                    end
                end

                @test !isnothing(problem)

                # train it ...
                p_net = FMIFlux.params(problem)
                
                solutionBefore = problem(X0; p = p_net, saveat = tData)
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
                    #printStep=true,
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

                # fig = plot(solutionAfter; title="Net $(i) - $(FAILED_GRADIENTS) / $(FAILED_GRADIENTS_QUOTA * NUMSTEPS)")
                # plot!(fig, tData, posData)
                # display(fig)
          end
       end
   end
end

@test length(fmu.components) <= 1

unloadFMU(fmu)
