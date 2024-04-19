# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons, Johannes Stoljar
# Licensed under the MIT license. 
# See LICENSE (https://github.com/thummeto/FMIFlux.jl/blob/main/LICENSE) file in the project root for details.

# imports
using FMI
using FMI.FMIImport: fmi2StringToValueReference, fmi2ValueReference, fmi2Real
using FMIFlux
using FMIFlux.Flux
using FMIZoo
using FMI.DifferentialEquations: Tsit5
using Statistics: mean, std
using Plots

# set seed
import Random
Random.seed!(1234);

tStart = 0.0
tStep = 0.1
tStop = 5.0
tSave = collect(tStart:tStep:tStop)

fricFMU = fmiLoad("SpringFrictionPendulum1D", "Dymola", "2022x")
fmiInfo(fricFMU)

vrs = ["mass.s", "mass.v", "mass.a", "mass.f"]
solFric = fmiSimulate(fricFMU, (tStart, tStop); recordValues=vrs, saveat=tSave)
plot(solFric)

posFric = fmi2GetSolutionValue(solFric, "mass.s")
velFric = fmi2GetSolutionValue(solFric, "mass.v")

x₀ = [posFric[1], velFric[1]]

fmiUnload(fricFMU)

simpleFMU = fmiLoad("SpringPendulum1D", "Dymola", "2022x"; type=:ME)
fmiInfo(simpleFMU)

vrs = ["mass.s", "mass.v", "mass.a"]
solSimple = fmiSimulate(simpleFMU, (tStart, tStop); recordValues=vrs, saveat=tSave)
plot(solSimple)

posSimple = fmi2GetSolutionValue(solSimple, "mass.s")
velSimple = fmi2GetSolutionValue(solSimple, "mass.v")

# loss function for training
global horizon = 5
function lossSum(p)
    global posFric, neuralFMU, horizon

    solution = neuralFMU(x₀, (tSave[1], tSave[horizon]); p=p, saveat=tSave[1:horizon]) # here, the NeuralODE is solved only for the time horizon

    posNet = fmi2GetSolutionState(solution, 1; isIndex=true)

    FMIFlux.Losses.mse(posFric[1:horizon], posNet)
end

function plotResults()
    global neuralFMU
    solNeural = neuralFMU(x₀, (tStart, tStop); saveat=tSave)
    
    fig = Plots.plot(xlabel="t [s]", ylabel="mass position [m]", linewidth=2,
                     xtickfontsize=12, ytickfontsize=12,
                     xguidefontsize=12, yguidefontsize=12,
                     legendfontsize=8, legend=:topright)
    
    plot!(fig, solSimple; stateIndices=1:1, values=false, label="SimpleFMU", linewidth=2)
    plot!(fig, solFric; valueIndices=1:1, label="FricFMU", linewidth=2)
    plot!(fig, solNeural; stateIndices=1:1, label="NeuralFMU", linewidth=2)
    fig
end

# callback function for training
global counter = 0
function callb(p)
    global counter, horizon 
    counter += 1
   
    if counter % 50 == 1
        avgLoss = lossSum(p[1])
        @info "  Loss [$counter] for horizon $horizon : $(round(avgLoss, digits=5))\nAvg displacement in data: $(round(sqrt(avgLoss), digits=5))"
        
        if avgLoss <= 0.01
            horizon += 2
            horizon = min(length(tSave), horizon)
        end
   
        # fig = plotResults()
        # println("Figure update.")
        # display(fig)
    end
end


global meanVal = 0.0
global stdVal = 0.0

function preProc!(data)
    global meanVal, stdVal

    meanVal = mean(data)
    stdVal = std(data)
    
    (data .- meanVal) ./ stdVal    
end 

function postProc!(data)
    global meanVal, stdVal
    
    (data .* stdVal) .+ meanVal
end 

# NeuralFMU setup
numStates = fmiGetNumberOfStates(simpleFMU)
additionalVRs = [fmi2StringToValueReference(simpleFMU, "mass.m")]
numAdditionalVRs = length(additionalVRs)

net = Chain(
    x -> simpleFMU(x=x, dx_refs=:all, y_refs=additionalVRs),
    preProc!,
    Dense(numStates+numAdditionalVRs, 16, tanh),
    postProc!,
    preProc!,
    Dense(16, 16, tanh),
    postProc!,
    preProc!,
    Dense(16, numStates),
    postProc!,
)

neuralFMU = ME_NeuralFMU(simpleFMU, net, (tStart, tStop), Tsit5(); saveat=tSave);

solutionBefore = neuralFMU(x₀)
fmiPlot(solutionBefore)

# train
paramsNet = Flux.params(neuralFMU)

optim = Adam()
FMIFlux.train!(lossSum, neuralFMU, Iterators.repeated((), 1000), optim; cb=()->callb(paramsNet)) 

# plot results mass.s
plotResults()

fmiUnload(simpleFMU)
