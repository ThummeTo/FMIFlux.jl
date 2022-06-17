#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons, Johannes Stoljar
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

################################## INSTALLATION ####################################################
# (1) Enter Package Manager via      ]
# (2) Install FMI via                add FMI       or   add "https://github.com/ThummeTo/FMI.jl"
# (3) Install FMIFlux via            add FMIFlux   or   add "https://github.com/ThummeTo/FMIFlux.jl"
# (4) Install FMIZoo via             add FMIZoo    or   add "https://github.com/ThummeTo/FMIZoo.jl"
# (5) Install Flux via               add Flux
# (6) Install DifferentialEquations  add DifferentialEquations
# (7) Install Plots via              add Plots
# (8) Install Random via             add Random
################################ END INSTALLATION ##################################################

# this example covers creation and training of a ME-NeuralFMU
using FMI
using FMI.FMIImport: fmi2StringToValueReference, fmi2ValueReference, fmi2Real
using FMIFlux
using FMIZoo
using Flux
using DifferentialEquations: Tsit5
#import Plots

# set seed
import Random
Random.seed!(1234);

tStart = 0.0
tStep = 0.1
tStop = 5.0
tSave = collect(tStart:tStep:tStop)

realFMU = fmiLoad("SpringFrictionPendulum1D", "Dymola", "2022x")
fmiInstantiate!(realFMU; loggingOn=false)
fmiSetupExperiment(realFMU, tStart, tStop)

fmiEnterInitializationMode(realFMU)
fmiExitInitializationMode(realFMU)

x₀ = fmiGetContinuousStates(realFMU)

vrs = ["mass.s", "mass.v", "mass.a", "mass.f"]
solution = fmiSimulate(realFMU, tStart, tStop; recordValues=vrs, saveat=tSave, reset=false)
#fmiPlot(solution)

realSimData = solution.values.saveval
posReal = collect(data[1] for data in realSimData)
velReal = collect(data[2] for data in realSimData)
fmiUnload(realFMU)

simpleFMU = fmiLoad("SpringPendulum1D", "Dymola", "2022x")
fmiInstantiate!(simpleFMU; loggingOn=false)
solution = fmiSimulate(simpleFMU, tStart, tStop; recordValues=vrs[1:end-1], saveat=tSave, reset=false)
simpleSimData = solution.values.saveval
posSimple = collect(data[1] for data in simpleSimData)


# loss function for training
function lossSum()
    global neuralFMU

    solution = neuralFMU(x₀)

    posNet = collect(data[1] for data in solution.states.u)
    # velNet = collect(data[2] for data in solution.states.u)

    Flux.mse(posReal, posNet)
end

# callback function for training
global counter = 0
function callb()
    global counter += 1
   
    if counter % 30 == 1
        avgLoss = lossSum()
        @info "   Loss [$counter]: $(round(avgLoss, digits=5))   
        Avg displacement in data: $(round(sqrt(avgLoss), digits=5))"
   
        fig = plotResults()
        println("Fig. update.")
        display(fig)
    end
end

function plotResults()
    solutionAfter = neuralFMU(x₀, tStart)
    fig = Plots.plot(xlabel="t [s]", ylabel="mass position [m]", linewidth=2,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12, legend=:topright)

    t = solutionAfter.states.t
    posNeuralFMU = collect(data[1] for data in solutionAfter.states.u)

    Plots.plot!(fig, tSave, posSimple, label="SimpleFMU", linewidth=2)
    Plots.plot!(fig, tSave, posReal, label="RealFMU", linewidth=2)
    Plots.plot!(fig, t, posNeuralFMU, label="NeuralFMU", linewidth=2)
    fig
end

# NeuralFMU setup
numStates = fmiGetNumberOfStates(simpleFMU)
additionalVRs = [fmi2StringToValueReference(simpleFMU, "mass.m")]
numAdditionalVRs = length(additionalVRs)

net = Chain(inputs -> fmiEvaluateME(simpleFMU, inputs, -1.0, zeros(fmi2ValueReference, 0), zeros(fmi2Real, 0), additionalVRs), 
            Dense(numStates+numAdditionalVRs, 16, tanh), 
            Dense(16, 16, tanh),
            Dense(16, numStates))

neuralFMU = ME_NeuralFMU(simpleFMU, net, (tStart, tStop), Tsit5(); saveat=tSave, convertParams=true)
solutionBefore = neuralFMU(x₀, tStart)
#fmiPlot(solutionBefore)

# train it ...
paramsNet = Flux.params(neuralFMU)

optim = ADAM()
# Feel free to increase training steps or epochs for better results
Flux.train!(lossSum, paramsNet, Iterators.repeated((), 1000), optim; cb=callb)

solutionAfter = neuralFMU(x₀, tStart)

fig = Plots.plot(xlabel="t [s]", ylabel="mass position [m]", linewidth=2,
                 xtickfontsize=12, ytickfontsize=12,
                 xguidefontsize=12, yguidefontsize=12,
                 legendfontsize=8, legend=:topright)

posNeuralFMU = collect(data[1] for data in solutionAfter.states.u)

Plots.plot!(fig, tSave, posSimple, label="SimpleFMU", linewidth=2)
Plots.plot!(fig, tSave, posReal, label="RealFMU", linewidth=2)
Plots.plot!(fig, tSave, posNeuralFMU, label="NeuralFMU (1000 epochs)", linewidth=2)
fig 

fmiUnload(simpleFMU)
