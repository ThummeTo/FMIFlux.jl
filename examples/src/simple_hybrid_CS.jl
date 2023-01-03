# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons, Johannes Stoljar
# Licensed under the MIT license. 
# See LICENSE (https://github.com/thummeto/FMIFlux.jl/blob/main/LICENSE) file in the project root for details.

# imports
using FMI
using FMIFlux
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
    solution = csNeuralFMU(extForce, tStep; p=p)

    accNet = fmi2GetSolutionValue(solution, 1; isIndex=true)
    
    FMIFlux.Losses.mse(accReference, accNet)
end

# callback function for training
global counter = 0
function callb(p)
    global counter += 1

    if counter % 20 == 1
        avgLoss = lossSum(p[1])
        @info "Loss [$counter]: $(round(avgLoss, digits=5))"
    end
end

# NeuralFMU setup
numInputs = length(defaultFMU.modelDescription.inputValueReferences)
numOutputs = length(defaultFMU.modelDescription.outputValueReferences)

function eval(u)
    y, _ = defaultFMU(;u_refs=defaultFMU.modelDescription.inputValueReferences, u=u, y_refs=defaultFMU.modelDescription.outputValueReferences)
    return y
end


net = Chain(inputs -> eval(inputs),
            Dense(numOutputs, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numOutputs))

csNeuralFMU = CS_NeuralFMU(defaultFMU, net, (tStart, tStop); saveat=tSave);

solutionBefore = csNeuralFMU(extForce, tStep)
accNeuralFMU = fmi2GetSolutionValue(solutionBefore, 1; isIndex=true)
Plots.plot(tSave, accNeuralFMU, label="acc CS-NeuralFMU", linewidth=2)

# train
paramsNet = FMIFlux.params(csNeuralFMU)

optim = Adam()
FMIFlux.train!(lossSum, paramsNet, Iterators.repeated((), 300), optim; cb=()->callb(paramsNet))

# plot results mass.a
solutionAfter = csNeuralFMU(extForce, tStep)

fig = Plots.plot(xlabel="t [s]", ylabel="mass acceleration [m/s^2]", linewidth=2,
                 xtickfontsize=12, ytickfontsize=12,
                 xguidefontsize=12, yguidefontsize=12,
                 legendfontsize=8, legend=:topright)

accNeuralFMU = fmi2GetSolutionValue(solutionAfter, 1; isIndex=true)

Plots.plot!(fig, tSave, accDefault, label="defaultFMU", linewidth=2)
Plots.plot!(fig, tSave, accReference, label="referenceFMU", linewidth=2)
Plots.plot!(fig, tSave, accNeuralFMU, label="CS-NeuralFMU (300 eps.)", linewidth=2)
fig 

fmiUnload(defaultFMU)
