# imports
using FMI
using FMIFlux
using Flux
using DifferentialEquations: Tsit5
import Plots

pathFMU = joinpath(dirname(@__FILE__), "../model/SpringPendulumExtForce1D.fmu")
println("FMU path: ", pathFMU)

tStart = 0.0
tStep = 0.01
tStop = 5.0
tSave = tStart:tStep:tStop

referenceFMU = fmiLoad(pathFMU)
fmiInstantiate!(referenceFMU; loggingOn=false)
fmiInfo(referenceFMU)

fmiSetupExperiment(referenceFMU, tStart, tStop)
fmiSetReal(referenceFMU, "mass_s0", 1.3)   # increase amplitude, invert phase
fmiEnterInitializationMode(referenceFMU)
fmiExitInitializationMode(referenceFMU)

x₀ = fmiGetContinuousStates(referenceFMU)

vrs = ["mass.s", "mass.v", "mass.a"]
_, referenceSimData = fmiSimulate(referenceFMU, tStart, tStop; recordValues=vrs, setup=false, reset=false, saveat=tSave)
fmiPlot(referenceFMU, vrs, referenceSimData)

posReference = collect(data[1] for data in referenceSimData.saveval)
velReference = collect(data[2] for data in referenceSimData.saveval)
accReference = collect(data[3] for data in referenceSimData.saveval)

fmiReset(referenceFMU)
defaultFMU = referenceFMU

fmiSetupExperiment(defaultFMU, tStart, tStop)
fmiEnterInitializationMode(defaultFMU)
fmiExitInitializationMode(defaultFMU)

x₀ = fmiGetContinuousStates(defaultFMU)

_, defaultSimData = fmiSimulate(defaultFMU, tStart, tStop; recordValues=vrs, setup=false, reset=false, saveat=tSave)
fmiPlot(defaultFMU, vrs, defaultSimData)

posDefault = collect(data[1] for data in defaultSimData.saveval)
velDefault = collect(data[2] for data in defaultSimData.saveval)
accDefault = collect(data[3] for data in defaultSimData.saveval)

function extForce(t)
    return [0.0]
end 

# loss function for training
function lossSum()
    solution = csNeuralFMU(extForce, tStep)

    accNet = collect(data[1] for data in solution)
    
    Flux.Losses.mse(accReference, accNet)
end

# callback function for training
global counter = 0
function callb()
    global counter += 1

    if counter % 20 == 1
        avgLoss = lossSum()
        @info "Loss [$counter]: $(round(avgLoss, digits=5))"
    end
end

# NeuralFMU setup
numInputs = length(defaultFMU.modelDescription.inputValueReferences)
numOutputs = length(defaultFMU.modelDescription.outputValueReferences)

net = Chain(inputs -> fmiInputDoStepCSOutput(defaultFMU, tStep, inputs),
            Dense(numOutputs, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numOutputs))

csNeuralFMU = CS_NeuralFMU(defaultFMU, net, (tStart, tStop); saveat=tSave);

solutionBefore = csNeuralFMU(extForce, tStep)
Plots.plot(tSave, collect(data[1] for data in solutionBefore), label="acc CS-NeuralFMU", linewidth=2)

# train
paramsNet = Flux.params(csNeuralFMU)

optim = ADAM()
Flux.train!(lossSum, paramsNet, Iterators.repeated((), 300), optim; cb=callb)

# plot results mass.a
solutionAfter = csNeuralFMU(extForce, tStep)

fig = Plots.plot(xlabel="t [s]", ylabel="mass acceleration [m/s^2]", linewidth=2,
                 xtickfontsize=12, ytickfontsize=12,
                 xguidefontsize=12, yguidefontsize=12,
                 legendfontsize=8, legend=:topright)

accNeuralFMU = collect(data[1] for data in solutionAfter)

Plots.plot!(fig, tSave, accDefault, label="defaultFMU", linewidth=2)
Plots.plot!(fig, tSave, accReference, label="referenceFMU", linewidth=2)
Plots.plot!(fig, tSave, accNeuralFMU, label="CS-NeuralFMU (300 eps.)", linewidth=2)
fig 

fmiUnload(defaultFMU)
