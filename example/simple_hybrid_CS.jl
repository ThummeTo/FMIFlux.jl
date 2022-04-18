# imports
using FMI
using FMIFlux
using FMIZoo
using Flux
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
fmiInstantiate!(referenceFMU; loggingOn=false)
fmiInfo(referenceFMU)

fmiSetupExperiment(referenceFMU, tStart, tStop)
fmiSetReal(referenceFMU, "mass_s0", 1.3)   # increase amplitude, invert phase
fmiEnterInitializationMode(referenceFMU)
fmiExitInitializationMode(referenceFMU)

x₀ = fmiGetContinuousStates(referenceFMU)

parameter = Dict("mass_s0" => 1.3)
vrs = ["mass.s", "mass.v", "mass.a"]
solution = fmiSimulate(referenceFMU, tStart, tStop; parameters=parameter, recordValues=vrs, saveat=tSave, reset=false)
fmiPlot(solution)

referenceSimData = solution.values.saveval
posReference = collect(data[1] for data in referenceSimData)
velReference = collect(data[2] for data in referenceSimData)
accReference = collect(data[3] for data in referenceSimData)

fmiTerminate(referenceFMU)
fmiReset(referenceFMU)
defaultFMU = referenceFMU

fmiSetupExperiment(defaultFMU, tStart, tStop)
fmiEnterInitializationMode(defaultFMU)
fmiExitInitializationMode(defaultFMU)

x₀ = fmiGetContinuousStates(defaultFMU)

solution = fmiSimulate(defaultFMU, tStart, tStop; recordValues=vrs, saveat=tSave, reset=false)
fmiPlot(solution)

defaultSimData = solution.values.saveval
posDefault = collect(data[1] for data in defaultSimData)
velDefault = collect(data[2] for data in defaultSimData)
accDefault = collect(data[3] for data in defaultSimData)

function extForce(t)
    return [0.0]
end 

# loss function for training
function lossSum()
    solution = csNeuralFMU(extForce, tStep)

    accNet = collect(data[1] for data in solution.values.saveval)
    
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
Plots.plot(tSave, collect(data[1] for data in solutionBefore.values.saveval), label="acc CS-NeuralFMU", linewidth=2)

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

accNeuralFMU = collect(data[1] for data in solutionAfter.values.saveval)

Plots.plot!(fig, tSave, accDefault, label="defaultFMU", linewidth=2)
Plots.plot!(fig, tSave, accReference, label="referenceFMU", linewidth=2)
Plots.plot!(fig, tSave, accNeuralFMU, label="CS-NeuralFMU (300 eps.)", linewidth=2)
fig 

fmiUnload(defaultFMU)
