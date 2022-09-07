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

# generate traing data
myFMU = fmiLoad("SpringPendulumExtForce1D", ENV["EXPORTINGTOOL"], ENV["EXPORTINGVERSION"])
parameters = Dict("mass_s0" => 1.3)
realSimData = fmiSimulateCS(myFMU, t_start, t_stop; parameters=parameters, recordValues=["mass.a"], setup=false, reset=false, saveat=tData)
fmiUnload(myFMU)

# setup traing data
function extForce(t)
    return [sin(t), cos(t)]
end
accData = fmi2GetSolutionValue(realSimData, "mass.a")

# loss function for training
function losssum()
    solution = problem(extForce, t_step)

    accNet = fmi2GetSolutionValue(solution, 1; isIndex=true)

    Flux.Losses.mse(accNet, accData)
end

# callback function for training
global iterCB = 0
global lastLoss = 0.0
function callb()
    global iterCB += 1
    global lastLoss

    if iterCB == 1
        lastLoss = losssum()
    end

    if iterCB % 50 == 0
        loss = losssum()
        @info "Loss: $loss"
        @test loss < lastLoss   
        lastLoss = loss
    end
end

# Load FMUs
fmus = Vector{FMU2}()
for i in 1:2 # how many FMUs do you want?
    fmu = fmiLoad("SpringPendulumExtForce1D", ENV["EXPORTINGTOOL"], ENV["EXPORTINGVERSION"])
    push!(fmus, fmu)
end

# NeuralFMU setup
total_fmu_outdim = sum(map(x->length(x.modelDescription.outputValueReferences), fmus))

net = Chain(
    Parallel(
        vcat,
        inputs -> fmi2InputDoStepCSOutput(fmus[1], t_step, inputs[1:1]),
        inputs -> fmi2InputDoStepCSOutput(fmus[2], t_step, inputs[2:2])
    ),
    Dense(total_fmu_outdim, 16, tanh),
    Dense(16, 16, tanh),
    Dense(16, length(fmus[1].modelDescription.outputValueReferences)),
)

problem = CS_NeuralFMU(fmus, net, (t_start, t_stop); saveat=tData)
@test problem != nothing

solutionBefore = problem(extForce, t_step)

# train it ...
p_net = Flux.params(problem)

optim = Adam()
Flux.train!(losssum, p_net, Iterators.repeated((), 100), optim; cb=callb)

# check results
solutionAfter = problem(extForce, t_step)

fmiUnload(myFMU)
