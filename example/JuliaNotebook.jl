# imports
# using Pkg
# Pkg.activate("Your Env")
using FMI
using FMIFlux
using Flux
using DifferentialEquations: Tsit5
import Plots

modelFMUPath = joinpath(dirname(@__FILE__), "../model/SpringPendulum1D.fmu")
realFMUPath = joinpath(dirname(@__FILE__), "../model/SpringFrictionPendulum1D.fmu")


t_start = 0.0
t_step = 0.01
t_stop = 5.0
tData = collect(t_start:t_step:t_stop)

myFMU = fmiLoad(realFMUPath)
fmiInstantiate!(myFMU; loggingOn=false)
fmiSetupExperiment(myFMU, t_start, t_stop)

fmiEnterInitializationMode(myFMU)
fmiExitInitializationMode(myFMU)

x0 = fmiGetContinuousStates(myFMU)

vrs = ["mass.s", "mass.v", "mass.f", "mass.a"]
_, realSimData = fmiSimulate(myFMU, t_start, t_stop; recordValues=vrs, saveat=tData, setup=false, reset=false)
fmiUnload(myFMU)

fmiPlot(myFMU, vrs, realSimData)

myFMU = fmiLoad(modelFMUPath)

fmiInstantiate!(myFMU; loggingOn=false)
_, fmuSimData = fmiSimulate(myFMU, t_start, t_stop; recordValues=["mass.s", "mass.v", "mass.a"], saveat=tData)

posData = collect(data[1] for data in realSimData.saveval)
velData = collect(data[2] for data in realSimData.saveval)

# loss function for training
function losssum()
    solution = problem(x0, t_start)

    posNet = collect(data[1] for data in solution.u)
    #velNet = collect(data[2] for data in solution.u)

    Flux.Losses.mse(posData, posNet) #+ Flux.Losses.mse(velData, velNet)
end

# callback function for training
global iterCB = 0
function callb()
    global iterCB += 1

    if iterCB % 10 == 1
        avg_ls = losssum()
        @info "Loss [$iterCB]: $(round(avg_ls, digits=5))   Avg displacement in data: $(round(sqrt(avg_ls), digits=5))"
    end
end

# NeuralFMU setup
numStates = fmiGetNumberOfStates(myFMU)

net = Chain(inputs -> fmiDoStepME(myFMU, inputs),
            Dense(numStates, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numStates))

problem = ME_NeuralFMU(myFMU, net, (t_start, t_stop), Tsit5(); saveat=tData)
solutionBefore = problem(x0, t_start)
fmiPlot(myFMU, solutionBefore)

# train it ...
p_net = Flux.params(problem)

optim = ADAM()
Flux.train!(losssum, p_net, Iterators.repeated((), 300), optim; cb=callb) # Feel free to increase training steps or epochs for better results

###### plot results mass.s
solutionAfter = problem(x0, t_start)
fig = Plots.plot(xlabel="t [s]", ylabel="mass position [m]", linewidth=2,
    xtickfontsize=12, ytickfontsize=12,
    xguidefontsize=12, yguidefontsize=12,
    legendfontsize=12, legend=:bottomright)
Plots.plot!(fig, tData, collect(data[1] for data in fmuSimData.saveval), label="FMU", linewidth=2)
Plots.plot!(fig, tData, posData, label="reference", linewidth=2)
Plots.plot!(fig, tData, collect(data[2] for data in solutionAfter.u), label="NeuralFMU", linewidth=2)
Plots.savefig(fig, "exampleResult_s.pdf")
fig 

fmiUnload(myFMU)

