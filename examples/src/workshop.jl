# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. 
# See LICENSE (https://github.com/thummeto/FMIFlux.jl/blob/main/LICENSE) file in the project root for details.

# This workshop was held at the JuliaCon2023 @ MIT (Boston)

# Loading in the required libraries
using FMIFlux       # for NeuralFMUs
using FMI           # import FMUs into Julia 
using FMIZoo        # a collection of demo models, including the VLDM
using FMIFlux.Flux  # Machine Learning in Julia

using JLD2          # data format for saving/loading parameters

# plotting
using Plots        # default plotting framework
import Optim

# for interactive plotting
import PlotlyJS     # plotting (interactive)
Plots.plotlyjs()    # actiavte PlotlyJS as default plotting backend

# Let's fix the random seed to make our program determinsitic (ANN layers are initialized indeterminsitic otherwise)
import Random 
Random.seed!(1234)

import FMI.DifferentiableEigen, FMIFlux.ForwardDiff

include(joinpath(@__DIR__, "workshop_helpers.jl"))

# load our FMU (we take one from the FMIZoo.jl, exported with Dymola 2022x)
fmu = fmiLoad("VLDM", "Dymola", "2020x"; type=:ME, logLevel=:info)  # "Log everything that might be interesting!", default is `:warn`

# let's have a look on the model meta data
fmiInfo(fmu)

# load data from FMIZoo.jl, gather simulation parameters for FMU
data = FMIZoo.VLDM(:train) 
tStart = data.consumption_t[1]
tStop = data.consumption_t[end]
tSave = data.consumption_t

# have a look on the FMU parameters (these are the file paths to the characteristic maps)
display(data.params)

# let's run a simulation from `tStart` to `tStop`, use the parameters we just viewed for the simulation run
resultFMU = fmiSimulate(fmu, (tStart, tStop); parameters=data.params)
display(resultFMU)

fig = plot(resultFMU)                                                                        # Plot it, but this is a bit too much, so ...
fig = plot(resultFMU; stateIndices=6:6)                                                      # ... only plot the state #6 and ...
fig = plot(resultFMU; stateIndices=6:6, ylabel="Cumulative consumption [Ws]", label="FMU")   # ... add some helpful labels!

# further plot the (measurement) data values `consumption_val` and deviation between measurements `consumption_dev`
plot!(fig, data.cumconsumption_t, data.cumconsumption_val; label="Data", ribbon=data.cumconsumption_dev, fillalpha=0.3)

# variable we want to manipulate - why we are picking exactly these three is shown a few lines later ;-)
manipulatedDerVars = ["der(dynamics.accelerationCalculation.integrator.y)",
                      "der(dynamics.accelerationCalculation.limIntegrator.y)",
                      "der(result.integrator.y)"]
# alternative: manipulatedDerVars = fmu.modelDescription.derivativeValueReferences[4:6]

# reference simulation to record the derivatives 
resultFMU = fmiSimulate(fmu, (tStart, tStop), parameters=data.params, recordValues=:derivatives, recordEigenvalues=true, saveat=tSave) 
manipulatedDerVals = fmiGetSolutionValue(resultFMU, manipulatedDerVars)

# what happens without propper transformation between FMU- and ANN-domain?
plot(resultFMU.values.t, manipulatedDerVals[1,:][1]; label="vehicle velocity");
plot!(resultFMU.values.t, tanh.(manipulatedDerVals[1,:][1]); label="tanh(velocity)")

# setup shift/scale layers for pre-processing
preProcess = ShiftScale(manipulatedDerVals)

# check what it's doing now ...
testVals = collect(preProcess(collect(val[t] for val in manipulatedDerVals))[1] for t in 1:length(resultFMU.values.t))
plot(resultFMU.values.t, testVals; label="velocity (pre-processed)");
plot!(resultFMU.values.t, tanh.(testVals); label="tanh(velocity)")

# add some additional "buffer"
preProcess.scale[:] *= 0.25 

# and check again what it's doing now ...
testVals = collect(preProcess(collect(val[t] for val in manipulatedDerVals))[1] for t in 1:length(resultFMU.values.t))
plot(resultFMU.values.t, testVals; label="velocity (pre-processed)");
plot!(resultFMU.values.t, tanh.(testVals); label="tanh(velocity)")

# ... also check the consumption
testVals = collect(preProcess(collect(val[t] for val in manipulatedDerVals))[3] for t in 1:length(resultFMU.values.t))
plot(resultFMU.values.t, testVals; label="vehicle consumption (pre-processed)");
plot!(resultFMU.values.t, tanh.(testVals); label="tanh(consumption)")

# setup scale/shift layer (inverse transformation) for post-processing
# we don't an inverse transform for the entire preProcess, only for the 2nd element (acceleration)
postProcess = ScaleShift(preProcess; indices=2:3) 

# setup cache layers 
cache = CacheLayer()
cacheRetrieve = CacheRetrieveLayer(cache)

gates = ScaleSum([1.0, 1.0, 0.0, 0.0], [[1,3], [2,4]]) # signal from FMU (#1 = 1.0), signal from ANN (#2 = 0.0)

# setup the NeuralFMU topology
net = Chain(x -> fmu(; x=x),                    # take `x`, put it into the FMU, retrieve `dx`
            dx -> cache(dx),                    # cache `dx`
            dx -> dx[4:6],                      # forward only dx[4, 5, 6]
            preProcess,                         # pre-process `dx`
            Dense(3, 32, tanh),                 # Dense Layer 3 -> 32 with `tanh` activasion
            Dense(32, 2, tanh),                 # Dense Layer 32 -> 2 with `tanh` activasion 
            postProcess,                        # post process `dx`
            dx -> cacheRetrieve(5:6, dx),       # dynamics FMU | dynamics ANN
            gates,                              # compute resulting dx from ANN + FMU
            dx -> cacheRetrieve(1:4, dx))       # stack together: dx[1,2,3,4] from cache + dx[5:6] from ANN

# build NeuralFMU
neuralFMU = ME_NeuralFMU(fmu, net, (tStart, tStop); saveat=tSave)
neuralFMU.modifiedState = false # speed optimization (no ANN before the FMU)

# get start state vector from data (FMIZoo)
x0 = FMIZoo.getStateVector(data, tStart)

########

# simulate and plot the (uninitialized) NeuralFMU
resultNFMU_original = neuralFMU(x0, (tStart, tStop); parameters=data.params, showProgress=true) 
display(resultNFMU_original)

fig = plot(resultNFMU_original; stateIndices=5:5, label="NeuralFMU (original)", ylabel="velocity [m/s]")

# plot the original FMU and data
plot!(fig, resultFMU; stateIndices=5:5, values=false, stateEvents=false)
plot!(fig, data.speed_t, data.speed_val, label="Data")

fig = plot(resultNFMU_original; stateIndices=6:6, stateEvents=false, timeEvents=false, label="NeuralFMU (original)", ylabel="velocity [m/s]")
plot!(fig, resultFMU; stateIndices=6:6, values=false, stateEvents=false, timeEvents=false, label="FMU")
plot!(fig, data.cumconsumption_t, data.cumconsumption_val, label="Data")

# prepare training data (array of arrays required)
train_data = collect([d] for d in data.cumconsumption_val)
train_t = data.consumption_t 

# switch to a more efficient execution configuration, allocate only a single FMU instance, see:
# https://thummeto.github.io/FMI.jl/dev/features/#Execution-Configuration
fmu.executionConfig = FMI.FMIImport.FMU2_EXECUTION_CONFIGURATION_NOTHING
c, _ = FMIFlux.prepareSolveFMU(neuralFMU.fmu, nothing, neuralFMU.fmu.type, true, false, false, false, true, data.params; x0=x0)

# batch the data (time, targets), train only on model output index 6, plot batch elements
batch = batchDataSolution(neuralFMU, t -> FMIZoo.getStateVector(data, t), train_t, train_data;
    batchDuration=BATCHDUR, indicesModel=6:6, plot=false, parameters=data.params, recordEigenvalues=true, recordEigenvaluesSensitivity=:ForwardDiff, showProgress=true) # try `plot=true` to show the batch elements, try `showProgress=true` to display simulation progress

# limit the maximum number of solver steps to 1e5 and maximum simulation/training duration to 30s
solverKwargsTrain = Dict{Symbol, Any}(:maxiters => round(Int, 100*BATCHDUR*10), :dtmin => 1e-128) # , :dtmin => 1e-6) # , :max_execution_duration => 60.0)
# for dt=10.0s, this equals 10 000 steps per second and 

cumconsumption_scale = 1.0 / (max(data.cumconsumption_val...)-min(data.cumconsumption_val...))
min_eig = min(collect(min(resultFMU.eigenvalues.saveval[i]...) for i in 2:length(resultFMU.eigenvalues.saveval))...)
allowedStiffness = (min_eig, 0.0)
function lossFct(solution::FMI.FMU2Solution, _data=data, _LOSS=LOSS, _EIGENLOSS=EIGENLOSS, _STIFFNESSRATIO=STIFFNESSRATIO)

    if !solution.success
        return [Inf]
    end

    #speeds = fmiGetSolutionState(solution, 5; isIndex=true)
    cumconsumption = fmiGetSolutionState(solution, 6; isIndex=true)

    dt = 0.1

    ts = 1+round(Int, solution.states.t[1]/dt)
    te = 1+round(Int, solution.states.t[end]/dt)
    num = te-ts+1

    target_cumconsumption = _data.cumconsumption_val[ts:te]

    Δcumconsumption = abs.(target_cumconsumption .- cumconsumption)
    Δcumconsumption -= _data.cumconsumption_dev[ts:te]
    Δcumconsumption = collect(max(cumconsumption, 0.0) for cumconsumption in Δcumconsumption)

    if _LOSS == :MAE
        Δcumconsumption = sum(Δcumconsumption) / num
    elseif _LOSS == :MSE
        Δcumconsumption = sum(Δcumconsumption .^ 2) / num
    else
        @assert false, "unknown LOSS"
    end

    eigen_loss = nothing
    if _EIGENLOSS == :MAE
        eigen_loss = FMIFlux.Losses.stiffness_corridor(solution, _STIFFNESSRATIO .* allowedStiffness; lossFct=Flux.Losses.mae)
    elseif _EIGENLOSS == :MSE
        eigen_loss = FMIFlux.Losses.stiffness_corridor(solution, _STIFFNESSRATIO .* allowedStiffness; lossFct=Flux.Losses.mse)
    elseif _EIGENLOSS == :OFF
        eigenLoss = 0.0
    else
        @assert false, "unknown EIGEN LOSS: $(_EIGEN_LOSS)"
    end
    
    if _EIGENLOSS == :OFF
        return [Δcumconsumption * cumconsumption_scale]
    else
        return [Δcumconsumption * cumconsumption_scale, eigen_loss]
    end
end

# initialize a "worst error growth scheduler" (updates all batch losses, pick the batch element with largest error increase)
# apply the scheduler after every training step, plot the current status every 25 steps and update all batch element losses every 5 steps
#scheduler = LossAccumulationScheduler(neuralFMU, batch, lossFct; applyStep=1, plotStep=1, updateStep=1)
scheduler = nothing
if SCHEDULER == :Random
    scheduler = RandomScheduler(neuralFMU, batch; applyStep=1, plotStep=1)
elseif SCHEDULER == :Sequential
    scheduler = SequentialScheduler(neuralFMU, batch; applyStep=1, plotStep=1)
elseif SCHEDULER == :LossAccumulation
    scheduler = LossAccumulationScheduler(neuralFMU, batch, lossFct; applyStep=1, plotStep=1, updateStep=5)
else
    @assert false, "unknown SCHEDULER"
end
updateScheduler = () -> update!(scheduler)

# defines a loss for the entire batch (accumulate error of batch elements)
batch_loss = p -> FMIFlux.Losses.batch_loss(neuralFMU, batch; 
    showProgress=true, p=p, parameters=data.params, recordEigenvalues=true, update=true, lossFct=lossFct, logLoss=true, solverKwargsTrain...) # try `showProgress=true` to display simulation progress

# loss for training, take element from the worst element scheduler
loss = p -> FMIFlux.Losses.loss(neuralFMU, batch; 
    showProgress=true, p=p, parameters=data.params, recordEigenvalues=true, recordEigenvaluesType=ForwardDiff.Dual, lossFct=lossFct, batchIndex=scheduler.elementIndex, logLoss=true, solverKwargsTrain...) # try `showProgress=true` to display simulation progress

# gather the parameters from the NeuralFMU
_params = FMIFlux.params(neuralFMU)

# let's check the loss we are starting with ...
loss_before = batch_loss(_params[1])
checkLoss(true)

batchLen = length(batch)

# initialize the scheduler 
Random.seed!(1234)
initialize!(scheduler; parameters=data.params, p=_params[1], showProgress=false)

function gateCallback()
    @info "\nAcc. FMU-Gate: $(round(_params[1][end-3]*100; digits=2))% | ANN-Gate: $(round(_params[1][end-1]*100; digits=2))%\n" * 
            "Con. FMU-Gate: $(round(_params[1][end-2]*100; digits=2))% | ANN-Gate: $(round(_params[1][end  ]*100; digits=2))%"
end

optim = Adam(ETA, (BETA1, BETA2))

# we use ForwardDiff for gradinet determination, because the FMU throws multiple events per time instant (this is not supported by reverse mode AD)
# the chunk_size controls the nuber of forward evaluations of the model (the bigger, the less evaluations)
FMIFlux.train!(loss, _params, Iterators.repeated((), TRAINSTEPS), optim; gradient=:ForwardDiff, chunk_size=32, cb=[updateScheduler, gateCallback], multiObjective=true, proceed_on_assert=true) 
loss_after = batch_loss(_params[1])
checkLoss(true)
checkLoss(true;cyle=:test)

# save the parameters (so we can use them tomorrow again)
# paramsPath = joinpath(@__DIR__, "params_$(scheduler.step)steps.jld2")
# fmiSaveParameters(neuralFMU, paramsPath)

# switch back to the default execution configuration, see:
# https://thummeto.github.io/FMI.jl/dev/features/#Execution-Configuration
fmu.executionConfig = FMI.FMIImport.FMU2_EXECUTION_CONFIGURATION_NO_RESET
FMIFlux.finishSolveFMU(neuralFMU.fmu, c, false, true)

validate()

# clean-up
fmiUnload(fmu) 