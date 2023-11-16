# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons, Johannes Stoljar
# Licensed under the MIT license. 
# See LICENSE (https://github.com/thummeto/FMIFlux.jl/blob/main/LICENSE) file in the project root for details.

# imports
using FMI
using FMIFlux
using FMIFlux.Flux
using FMIZoo
using DifferentialEquations: Tsit5
import Plots

# set seed
import Random
Random.seed!(1234);

tStart = 0.0
tStep = 0.01
tStop = 5.0
tSave = tStart:tStep:tStop

referenceFMU = fmiLoad("SpringPendulumExtForce1D", "Dymola", "2022x")
fmiInfo(referenceFMU)

param = Dict("mass_s0" => 1.3, "mass.v" => 0.0)   # increase amplitude, invert phase
vrs = ["mass.s", "mass.v", "mass.a"]
referenceSimData = fmiSimulate(referenceFMU, (tStart, tStop); parameters=param, recordValues=vrs, saveat=tSave)
fmiPlot(referenceSimData)

posReference = fmi2GetSolutionValue(referenceSimData, vrs[1])
velReference = fmi2GetSolutionValue(referenceSimData, vrs[2])
accReference = fmi2GetSolutionValue(referenceSimData, vrs[3])

defaultFMU = referenceFMU
param = Dict("mass_s0" => 0.5, "mass.v" => 0.0)

defaultSimData = fmiSimulate(defaultFMU, (tStart, tStop); parameters=param, recordValues=vrs, saveat=tSave)
fmiPlot(defaultSimData)

posDefault = fmi2GetSolutionValue(defaultSimData, vrs[1])
velDefault = fmi2GetSolutionValue(defaultSimData, vrs[2])
accDefault = fmi2GetSolutionValue(defaultSimData, vrs[3])

function extForce(t)
    return [0.0]
end 

# loss function for training
function lossSum(p)
    solution = csNeuralFMU(extForce, tStep, (tStart, tStop); p=p) # saveat=tSave

    accNet = fmi2GetSolutionValue(solution, 2; isIndex=true)
    
    FMIFlux.Losses.mse(accReference, accNet)
end

# callback function for training
global counter = 0
function callb(p)
    global counter += 1

    if counter % 25 == 1
        avgLoss = lossSum(p[1])
        @info "Loss [$counter]: $(round(avgLoss, digits=5))"
    end
end

# check outputs
outputs = defaultFMU.modelDescription.outputValueReferences 
numOutputs = length(outputs)
display(outputs)

# check inputs
inputs = defaultFMU.modelDescription.inputValueReferences 
numInputs = length(inputs)
display(inputs)

# NeuralFMU setup
net = Chain(u -> defaultFMU(;u_refs=inputs, u=u, y_refs=outputs),
            Dense(numOutputs, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numOutputs))

csNeuralFMU = CS_NeuralFMU(defaultFMU, net, (tStart, tStop));

solutionBefore = csNeuralFMU(extForce, tStep, (tStart, tStop)) # ; saveat=tSave
accNeuralFMU = fmi2GetSolutionValue(solutionBefore, 1; isIndex=true)
Plots.plot(tSave, accNeuralFMU, label="acc CS-NeuralFMU", linewidth=2)

# train
paramsNet = FMIFlux.params(csNeuralFMU)

optim = Adam()
FMIFlux.train!(lossSum, paramsNet, Iterators.repeated((), 250), optim; cb=()->callb(paramsNet))

# plot results mass.a
solutionAfter = csNeuralFMU(extForce, tStep, (tStart, tStop)) # saveat=tSave, p=paramsNet[1]

fig = Plots.plot(xlabel="t [s]", ylabel="mass acceleration [m/s^2]", linewidth=2,
                 xtickfontsize=12, ytickfontsize=12,
                 xguidefontsize=12, yguidefontsize=12,
                 legendfontsize=8, legend=:topright)

accNeuralFMU = fmi2GetSolutionValue(solutionAfter, 2; isIndex=true)

Plots.plot!(fig, tSave, accDefault, label="defaultFMU", linewidth=2)
Plots.plot!(fig, tSave, accReference, label="referenceFMU", linewidth=2)
Plots.plot!(fig, tSave, accNeuralFMU, label="CS-NeuralFMU (1000 eps.)", linewidth=2)
fig 

fmiUnload(defaultFMU)
