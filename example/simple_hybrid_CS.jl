# imports
using FMI
using FMIFlux
using Flux
using DifferentialEquations: Tsit5
import Plots

fmuPath = joinpath(dirname(@__FILE__), "../model/SpringPendulumExtForce1D.fmu")
println("FMU path: ", fmuPath)

tStart = 0.0
tStep = 0.01
tStop = 5.0
tSave = tStart:tStep:tStop

referenceFmu = fmiLoad(fmuPath)
fmiInstantiate!(referenceFmu; loggingOn=false)
fmiInfo(referenceFmu)

fmiSetupExperiment(referenceFmu, tStart, tStop)
fmiSetReal(referenceFmu, "mass_s0", 1.3)   # increase amplitude, invert phase
fmiEnterInitializationMode(referenceFmu)
fmiExitInitializationMode(referenceFmu)

x₀ = fmiGetContinuousStates(referenceFmu)

vrs = ["mass.s", "mass.v", "mass.a"]
_, referenceSimData = fmiSimulate(referenceFmu, tStart, tStop; recordValues=vrs, setup=false, reset=false, saveat=tSave)
fmiPlot(referenceFmu, vrs, referenceSimData)

posReference = collect(data[1] for data in referenceSimData.saveval)
velReference = collect(data[2] for data in referenceSimData.saveval)
accReference = collect(data[3] for data in referenceSimData.saveval)

fmiReset(referenceFmu)
defaultFmu = referenceFmu

fmiSetupExperiment(defaultFmu, tStart, tStop)
fmiEnterInitializationMode(defaultFmu)
fmiExitInitializationMode(defaultFmu)

x₀ = fmiGetContinuousStates(defaultFmu)

_, defaultSimData = fmiSimulate(defaultFmu, tStart, tStop; recordValues=vrs, setup=false, reset=false, saveat=tSave)
fmiPlot(defaultFmu, vrs, defaultSimData)

posDefault = collect(data[1] for data in defaultSimData.saveval)
velDefault = collect(data[2] for data in defaultSimData.saveval)
accDefault = collect(data[3] for data in defaultSimData.saveval)

function extForce(t)
    return [0.0]
end 

# loss function for training
function lossSum()
    solution = csNeuralFmu(extForce, tStep)

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
numInputs = length(defaultFmu.modelDescription.inputValueReferences)
numOutputs = length(defaultFmu.modelDescription.outputValueReferences)

net = Chain(inputs -> fmiInputDoStepCSOutput(defaultFmu, tStep, inputs),
            Dense(numOutputs, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numOutputs))

csNeuralFmu = CS_NeuralFMU(defaultFmu, net, (tStart, tStop); saveat=tSave);

solutionBefore = csNeuralFmu(extForce, tStep)
Plots.plot(tSave, collect(data[1] for data in solutionBefore), label="acc CS-NeuralFMU", linewidth=2)

# train
paramsNet = Flux.params(csNeuralFmu)

optim = ADAM()
Flux.train!(lossSum, paramsNet, Iterators.repeated((), 300), optim; cb=callb)

# plot results mass.a
solutionAfter = csNeuralFmu(extForce, tStep)

fig = Plots.plot(xlabel="t [s]", ylabel="mass acceleration [m/s^2]", linewidth=2,
                 xtickfontsize=12, ytickfontsize=12,
                 xguidefontsize=12, yguidefontsize=12,
                 legendfontsize=8, legend=:topright)

accNeuralFmu = collect(data[1] for data in solutionAfter)

Plots.plot!(fig, tSave, accDefault, label="defaultFMU", linewidth=2)
Plots.plot!(fig, tSave, accReference, label="referenceFMU", linewidth=2)
Plots.plot!(fig, tSave, accNeuralFmu, label="CS-NeuralFMU (300 eps.)", linewidth=2)
fig 

fmiUnload(defaultFmu)
