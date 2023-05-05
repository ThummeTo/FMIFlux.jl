#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using FMI
using Flux
using DifferentialEquations: Tsit5

import Random 
Random.seed!(1234);

t_start = 0.0
t_step = 0.01
t_stop = 5.0
tData = t_start:t_step:t_stop

# generate training data
realFMU = fmiLoad("SpringFrictionPendulum1D", ENV["EXPORTINGTOOL"], ENV["EXPORTINGVERSION"]; type=fmi2TypeCoSimulation)
realSimData = fmiSimulateCS(realFMU, (t_start, t_stop); recordValues=["mass.s", "mass.v"], saveat=tData)
x0 = collect(realSimData.values.saveval[1])
@test x0 == [0.5, 0.0]

# load FMU for NeuralFMU
myFMU = fmiLoad("SpringPendulum1D", ENV["EXPORTINGTOOL"], ENV["EXPORTINGVERSION"]; type=fmi2TypeModelExchange)

using FMIImport
c = fmi2Instantiate!(myFMU)
fmi2SetupExperiment(c, 0.0, 1.0)
fmi2EnterInitializationMode(c)
fmi2ExitInitializationMode(c)
p_refs = myFMU.modelDescription.parameterValueReferences
p = fmi2GetReal(c, p_refs)

# setup traing data
velData = sin.(tData) # fmi2GetSolutionValue(realSimData, "mass.v")

# loss function for training
function losssum(p)
    global problem, x0, posData
    solution = problem(x0; p=p, showProgress=true)

    if !solution.success
        return Inf 
    end

    # posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    velNet = fmi2GetSolutionState(solution, 2; isIndex=true)
    
    return Flux.Losses.mse(velNet, velData) # Flux.Losses.mse(posNet, posData)
end

# callback function for training
global iterCB = 0
global lastLoss = 0.0
function callb(p)
    global iterCB += 1
    global lastLoss

    if iterCB % 5 == 0
        loss = losssum(p[1])
        @info "[$(iterCB)] Loss: $loss"
        @test loss < lastLoss  
        lastLoss = loss
    end
end

numStates = fmiGetNumberOfStates(myFMU)

using FMIImport.FMICore
import FMIImport: unsense

struct myFMULayer 
    p::AbstractArray{<:Real}

    function myFMULayer(p::AbstractArray{<:Real})
        inst = new(p)
        return inst
    end
end

function (l::myFMULayer)(x)
    global myFMU
    myFMU.optim_p = unsense(l.p)
    myFMU.optim_p_refs = p_refs

    dxy = myFMU(;x=x, p=l.p, p_refs=p_refs)

    return dxy
end

Flux.@functor myFMULayer (p,)

net = Chain(myFMULayer(p))

optim = Adam(1e-3)
solver = Tsit5()

problem = ME_NeuralFMU(myFMU, net, (t_start, t_stop), solver; saveat=tData)
@test problem != nothing

solutionBefore = problem(x0)
if solutionBefore.success
    @test length(solutionBefore.states.t) == length(tData)
    @test solutionBefore.states.t[1] == t_start
    @test solutionBefore.states.t[end] == t_stop
end

# train it ...
p_net = Flux.params(problem)
@test length(p_net) == 1
@test length(p_net[1]) == 7

iterCB = 0
lastLoss = losssum(p_net[1])
@info "Start-Loss for net: $lastLoss"
FMIFlux.train!(losssum, p_net, Iterators.repeated((), parse(Int, ENV["NUMSTEPS"])), optim; cb=()->callb(p_net))

# check results
solutionAfter = problem(x0)
if solutionAfter.success
    @test length(solutionAfter.states.t) == length(tData)
    @test solutionAfter.states.t[1] == t_start
    @test solutionAfter.states.t[end] == t_stop
end

@test length(myFMU.components) <= 1

fmiUnload(realFMU)
fmiUnload(myFMU)
