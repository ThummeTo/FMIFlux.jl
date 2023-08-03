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
t_stop = 1.0
tData = t_start:t_step:t_stop

# generate traing data
myFMU = fmiLoad("SpringPendulumExtForce1D", EXPORTINGTOOL, EXPORTINGVERSION; type=:CS)
parameters = Dict("mass_s0" => 1.3)
realSimData = fmiSimulateCS(myFMU, (t_start, t_stop); parameters=parameters, recordValues=["mass.a"], saveat=tData)
fmiUnload(myFMU)

# setup traing data
function extForce(t)
    return [sin(t), cos(t)]
end
accData = fmi2GetSolutionValue(realSimData, "mass.a")

# loss function for training
function losssum(p)
    solution = problem(extForce, t_step; p=p)

    if !solution.success
        return Inf 
    end

    accNet = fmi2GetSolutionValue(solution, 1; isIndex=true)

    Flux.Losses.mse(accNet, accData)
end

# callback function for training
global iterCB = 0
global lastLoss = 0.0
function callb(p)
    global iterCB += 1
    global lastLoss

    if iterCB == 1
        lastLoss = losssum(p[1])
    end

    if iterCB % 5 == 0
        loss = losssum(p[1])
        @info "[$(iterCB)] Loss: $loss"
        @test loss < lastLoss   
        lastLoss = loss
    end
end

# Load FMUs
fmus = Vector{FMU2}()
for i in 1:2 # how many FMUs do you want?
    _fmu = fmiLoad("SpringPendulumExtForce1D", EXPORTINGTOOL, EXPORTINGVERSION; type=fmi2TypeCoSimulation)
    push!(fmus, _fmu)
end

# NeuralFMU setup
total_fmu_outdim = sum(map(x->length(x.modelDescription.outputValueReferences), fmus))

function eval(i, u)
    fmus[i](;u_refs=fmus[i].modelDescription.inputValueReferences, u=u, y_refs=fmus[i].modelDescription.outputValueReferences)
end
net = Chain(
    Parallel(
        vcat,
        inputs -> eval(1, inputs[1:1]),
        inputs -> eval(2, inputs[2:2])
    ),
    Dense(total_fmu_outdim, 16, tanh; init=Flux.identity_init),
    Dense(16, 16, tanh; init=Flux.identity_init),
    Dense(16, length(fmus[1].modelDescription.outputValueReferences); init=Flux.identity_init),
)

problem = CS_NeuralFMU(fmus, net, (t_start, t_stop); saveat=tData)
@test problem != nothing

solutionBefore = problem(extForce, t_step)

# train it ...
p_net = Flux.params(problem)

optim = Adam(1e-6)

# ToDo: In CS-Mode, each training step takes longer than the previuous one... this is a very strange behaviour.
# Because this can only be cured by restarting Julia (not by reevaluation of code/constructors), this may be a error somewhere deeper than in FMIFlux.jl 
FMIFlux.train!(losssum, p_net, Iterators.repeated((), 1), optim; cb=()->callb(p_net))

# check results
solutionAfter = problem(extForce, t_step)

for i in 1:length(fmus)
    fmiUnload(fmus[i])
end
