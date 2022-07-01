# imports
using FMI
using FMI.FMIImport: fmi2StringToValueReference, fmi2ValueReference, fmi2Real
using FMIFlux
using FMIZoo
using Flux
using DifferentialEquations: Tsit5
using Statistics: mean, std
import Plots

# set seed
import Random
Random.seed!(1234);

tStart = 0.0
tStep = 0.1
tStop = 5.0
tSave = collect(tStart:tStep:tStop)

realFMU = fmiLoad("SpringFrictionPendulum1D", "Dymola", "2022x")
fmiInfo(realFMU)

vrs = ["mass.s", "mass.v", "mass.a", "mass.f"]
realSimData = fmiSimulate(realFMU, tStart, tStop; recordValues=vrs, saveat=tSave)
fmiPlot(realSimData)

posReal = fmi2GetSolutionValue(realSimData, "mass.s")
velReal = fmi2GetSolutionValue(realSimData, "mass.v")

x₀ = [posReal[1], velReal[1]]

fmiUnload(realFMU)

simpleFMU = fmiLoad("SpringPendulum1D", "Dymola", "2022x")
fmiInfo(simpleFMU)

vrs = ["mass.s", "mass.v", "mass.a"]
simpleSimData = fmiSimulate(simpleFMU, tStart, tStop; recordValues=vrs, saveat=tSave)
fmiPlot(simpleSimData)

posSimple = fmi2GetSolutionValue(simpleSimData, "mass.s")
velSimple = fmi2GetSolutionValue(simpleSimData, "mass.v")

# loss function for training
global horizon = 5
function lossSum()
    global posReal, neuralFMU, horizon
    solution = neuralFMU(x₀, tStart)

    posNet = fmi2GetSolutionState(solution, 1; isIndex=true)
    
    horizon = min(length(posNet), horizon)

    Flux.Losses.mse(posReal[1:horizon], posNet[1:horizon])
end

function plotResults()
    global neuralFMU
    solution = neuralFMU(x₀, tStart)

    posNeural = fmi2GetSolutionState(solution, 1; isIndex=true)
    time = fmi2GetSolutionTime(solution)
    
    fig = Plots.plot(xlabel="t [s]", ylabel="mass position [m]", linewidth=2,
                     xtickfontsize=12, ytickfontsize=12,
                     xguidefontsize=12, yguidefontsize=12,
                     legendfontsize=8, legend=:topright)
    
    Plots.plot!(fig, tSave, posSimple, label="SimpleFMU", linewidth=2)
    Plots.plot!(fig, tSave, posReal, label="RealFMU", linewidth=2)
    Plots.plot!(fig, time, posNeural, label="NeuralFMU", linewidth=2)
    fig
end

# callback function for training
global counter = 0
function callb()
    global counter, horizon 
    counter += 1
   
    if counter % 20 == 1
        avgLoss = lossSum()
        @info "   Loss [$counter] for horizon $horizon : $(round(avgLoss, digits=5))   
        Avg displacement in data: $(round(sqrt(avgLoss), digits=5))"
        
        if avgLoss <= 0.01
            horizon += 2
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
    inputs -> fmiEvaluateME(simpleFMU, inputs, -1.0, zeros(fmi2ValueReference, 0), 
                            zeros(fmi2Real, 0), additionalVRs),
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

solutionBefore = neuralFMU(x₀, tStart)
fmiPlot(solutionBefore)

# train
paramsNet = Flux.params(neuralFMU)

optim = ADAM()
Flux.train!(lossSum, paramsNet, Iterators.repeated((), 1000), optim; cb=callb) 

# plot results mass.s
plotResults()

fmiUnload(simpleFMU)
