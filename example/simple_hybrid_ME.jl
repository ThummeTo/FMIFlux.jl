# imports
using FMI
using FMIFlux
using FMIZoo
using Flux
using DifferentialEquations: Tsit5
import Plots

# set seed
import Random
Random.seed!(42);

tStart = 0.0
tStep = 0.01
tStop = 5.0
tSave = collect(tStart:tStep:tStop)

realFMU = fmiLoad("SpringFrictionPendulum1D", "Dymola", "2022x")
fmiInstantiate!(realFMU; loggingOn=false)
fmiInfo(realFMU)

fmiSetupExperiment(realFMU, tStart, tStop)

fmiEnterInitializationMode(realFMU)
fmiExitInitializationMode(realFMU)

x₀ = fmiGetContinuousStates(realFMU)

vrs = ["mass.s", "mass.v", "mass.a", "mass.f"]
solution = fmiSimulate(realFMU, tStart, tStop; recordValues=vrs, saveat=tSave, reset=false)
fmiPlot(solution)

fmiUnload(realFMU)

realSimData = solution.values.saveval
velReal = collect(data[2] for data in realSimData)
posReal = collect(data[1] for data in realSimData)

simpleFMU = fmiLoad("SpringPendulum1D", "Dymola", "2022x")

fmiInstantiate!(simpleFMU; loggingOn=false)
fmiInfo(simpleFMU)

vrs = ["mass.s", "mass.v", "mass.a"]
solution = fmiSimulate(simpleFMU, tStart, tStop; recordValues=vrs, saveat=tSave, reset=false)
fmiPlot(solution)

simpleSimData = solution.values.saveval
velSimple = collect(data[2] for data in simpleSimData)
posSimple = collect(data[1] for data in simpleSimData)

# loss function for training
function lossSum()
    solution = neuralFMU(x₀, tStart)

    posNet = collect(data[1] for data in solution.states.u)
    #velNet = collect(data[2] for data in solution.states.u)

    Flux.Losses.mse(posReal, posNet) #+ Flux.Losses.mse(velReal, velNet)
end

# callback function for training
global counter = 0
function callb()
    global counter += 1
    if counter % 20 == 1
        avgLoss = lossSum()
        @info "Loss [$counter]: $(round(avgLoss, digits=5))   Avg displacement in data: $(round(sqrt(avgLoss), digits=5))"
    end
end

# NeuralFMU setup
numStates = fmiGetNumberOfStates(simpleFMU)

net = Chain(inputs -> fmiEvaluateME(simpleFMU, inputs),
            Dense(numStates, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numStates))

neuralFMU = ME_NeuralFMU(simpleFMU, net, (tStart, tStop), Tsit5(); saveat=tSave);

solutionBefore = neuralFMU(x₀, tStart)
fmiPlot(solutionBefore)

# train
paramsNet = Flux.params(neuralFMU)

optim = ADAM()
Flux.train!(lossSum, paramsNet, Iterators.repeated((), 300), optim; cb=callb) 

# plot results mass.s
solutionAfter = neuralFMU(x₀, tStart)

fig = Plots.plot(xlabel="t [s]", ylabel="mass position [m]", linewidth=2,
                 xtickfontsize=12, ytickfontsize=12,
                 xguidefontsize=12, yguidefontsize=12,
                 legendfontsize=8, legend=:topright)

posNeuralFMU = collect(data[1] for data in solutionAfter.states.u)

Plots.plot!(fig, tSave, posSimple, label="SimpleFMU", linewidth=2)
Plots.plot!(fig, tSave, posReal, label="RealFMU", linewidth=2)
Plots.plot!(fig, tSave, posNeuralFMU, label="NeuralFMU (300 epochs)", linewidth=2)
fig 

Flux.train!(lossSum, paramsNet, Iterators.repeated((), 700), optim; cb=callb) 
# plot results mass.s
solutionAfter = neuralFMU(x₀, tStart)
posNeuralFMU = collect(data[1] for data in solutionAfter.states.u)
Plots.plot!(fig, tSave, posNeuralFMU, label="NeuralFMU (1000 epochs)", linewidth=2)
fig 

fmiUnload(simpleFMU)
