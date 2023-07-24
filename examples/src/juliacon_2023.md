# Using NeuralODEs in real life applications
-----
Tutorial by Tobias Thummerer | Last edit: 07-21-2023

This workshop was held at the JuliaCon 2023 | 07-25-2023 | MIT (Boston, USA)

Keywords: *#NeuralODE, #NeuralFMU, #PeNODE, #HybridModeling*

## Introduction
NeuralODEs lead to amazing results in academic examples. But the expectations are often being disappointed as soon as one tries to adapt this concept for real life use cases. Bad convergence behavior, handling of discontinuities and/or instabilities are just some of the stumbling blocks that might pop up during the first steps. During the workshop, we want to show how to integrate real life industrial models in NeuralODEs using FMI and present sophisticated training strategies.

This tutorial can be used in two ways:
1. As a single script, showing how a NeuralFMU can be setup and trained. Results can be loaded from a precomputed hyperparameter optimization.
2. As a module (see sections *Optional: Organize as module*) together with the file `juliacon_2023_distributedhyperopt.jl` to perform your own distributed hyperparameter optimization.

## License


```julia
# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. 
# See LICENSE (https://github.com/thummeto/FMIFlux.jl/blob/main/LICENSE) file in the project root for details.

# This workshop was held at the JuliaCon2023 @ MIT (Boston)
```

# Optional: Organize as module
If you want, you can place all code inside of a module named `NODE_Training`, this simplifies hyper parameter optimization.


```julia
# for hyper parameter optimization, place the code in a `module`
# just uncomment the following lines (and the one at the very end, too!) 
#module NODE_Training 
#using DistributedHyperOpt
#using DistributedHyperOpt.Distributed
```

## 1. Loading the Packages
Before we start modeling our NeuralODE, we load all required packages. If some packages are still missing, install them by typing `import Pkg; Pkg.add("[PKG-NAME]")`.


```julia
# Loading the required libraries
using FMIFlux       # for NeuralFMUs
using FMI           # import FMUs into Julia 
using FMIZoo        # a collection of demo models, including the VLDM
using FMIFlux.Flux  # Machine Learning in Julia

import JLD2         # data format for saving/loading parameters

import Random       # for fixing the random seed
using Plots         # plotting results

# A variable indicating the hyperparameter run
HPRUN = 0 
```

Beside the packages, we use another little script that includes some nice plotting functions specialy for this workshop.


```julia
# a helper file with some predefined functions to make "things look nicer", but are not really relevant to the topic
include(joinpath(@__DIR__, "juliacon_2023_helpers.jl"));
```

Because notebooks can't handle progress bars, we disable *progress bar printing* - but feel free to enable it if you are using the code outside of a jupyter notebook. The progress bar gives further helpful information, like the estimated remaining computation time for simulation and training.


```julia
# disable progress bars in jupyter notebook
showProgress=false;
```

## 2. Loading FMU & Data
Before starting with hybrid modeling, we load in the used training data and our FMU of the VLDM. We simulate the FMU, plot the results and compare them to data. 

We start by loading in the data (training and validation) used in this tutorial from *FMIZoo.jl* - a container library for different system model FMUs and corresponding data.


```julia
# sample length for data is 0.1s 
dt = 0.1 

# load data (training and validation) from FMIZoo.jl, gather simulation parameters for FMU
data = VLDM(:train, dt=dt) 
data_validation = VLDM(:validate, dt=dt)

# start (`tStart`) and stop time (`tStop`) for simulation, saving time points for ODE solver (`tSave`)
tStart = data.consumption_t[1]
tStop = data.consumption_t[end]
tSave = data.consumption_t

# have a look on the FMU parameters (these are the file paths to the characteristic maps, remaining parameters are set to default by the FMU)
display(data.params)
```

After that, we load the FMU and have a look on its model meta data.


```julia
# load our FMU of the VLDM (we take it from the FMIZoo.jl, exported with Dymola 2020x)
fmu = fmiLoad("VLDM", "Dymola", "2020x"; type=:ME, logLevel=:info)  # "Log everything that might be interesting!", default is `:warn`

# let's have a look on the model meta data
fmiInfo(fmu)
```

Simulating is as easy as calling `fmiSimulate`. Note, that we are putting in the paramter dictionary `data.params` from above. This FMU has many events, these are detected and handled automatically by *FMI.jl*.


```julia
# let's run a simulation from `tStart` to `tStop`, use the parameters we just viewed for the simulation run
resultFMU = fmiSimulate(fmu, (tStart, tStop); parameters=data.params, showProgress=showProgress)
display(resultFMU)

# Plot the simulation results
fig = plot(resultFMU)                                                                        # Plot it, but this is a bit too much, so ...
fig = plot(resultFMU; stateIndices=6:6)                                                      # ... only plot the state #6 (the cumulative consumption) and ...
fig = plot(resultFMU; stateIndices=6:6, ylabel="Cumulative consumption [Ws]", label="FMU")   # ... add some helpful labels!

# further plot the (measurement) data values `consumption_val` and deviation between measurements `consumption_dev`
plot!(fig, data.cumconsumption_t, data.cumconsumption_val; label="Data", ribbon=data.cumconsumption_dev, fillalpha=0.3)
```

We do another simulation run and record the derivative values, to have a good starting point for scaling the pre- and post-processing layers.


```julia
# variable we want to manipulate - why we are picking exactly these three is shown a few lines later ;-)
manipulatedDerVars = ["der(dynamics.accelerationCalculation.integrator.y)",
                      "der(dynamics.accelerationCalculation.limIntegrator.y)",
                      "der(result.integrator.y)"]
# alternative: manipulatedDerVars = fmu.modelDescription.derivativeValueReferences[4:6]

# reference simulation to record the derivatives 
resultFMU = fmiSimulate(fmu, (tStart, tStop), parameters=data.params, recordValues=:derivatives, saveat=tSave, showProgress=showProgress) 
manipulatedDerVals = fmiGetSolutionValue(resultFMU, manipulatedDerVars)

# what happens without propper transformation between FMU- and ANN-domain?
plot(resultFMU.values.t, manipulatedDerVals[1,:][1]; label="vehicle velocity [m/s]");
plot!(resultFMU.values.t, tanh.(manipulatedDerVals[1,:][1]); label="tanh(velocity)")
```

That's why we need shift- and scale layers!

Finally, the FMU is unloaded again to release memory.


```julia
# unload FMU
fmiUnload(fmu)
```

## 3. NeuralFMU setup

![NeuralFMU](https://github.com/thummeto/FMIFlux.jl/blob/main/docs/src/examples/img/juliacon_2023/neuralfmu_topology.png?raw=true)

Equipped with data and a simulation model, we can setup the NeuralFMU as introduced in the workshop.


```julia
# function that builds the considered NeuralFMU on basis of a given FMU (FMI-Version 2.0) `f`
function build_NFMU(f::FMU2)
    
    # pre- and post-processing
    preProcess = ShiftScale(manipulatedDerVals)
    preProcess.scale[:] *= 0.25 # add some additional "buffer"
    postProcess = ScaleShift(preProcess; indices=2:3) 

    # cache
    cache = CacheLayer()
    cacheRetrieve = CacheRetrieveLayer(cache)

    # the gates layer, signals are:
    # (1) acceleration from FMU (gate=1.0 | open)
    # (2) consumption  from FMU (gate=1.0 | open)
    # (3) acceleration from ANN (gate=0.0 | closed)
    # (4) consumption  from ANN (gate=0.0 | closed)
    # the acelerations [1,3] and consumptions [2,4] are paired
    gates = ScaleSum([1.0, 1.0, 0.0, 0.0], [[1,3], [2,4]]) 

    # setup the NeuralFMU topology
    net = Chain(x -> f(; x=x),                  # take `x`, put it into the FMU, retrieve `dx`
            dx -> cache(dx),                    # cache `dx`
            dx -> dx[4:6],                      # forward only dx[4, 5, 6]
            preProcess,                         # pre-process `dx`
            Dense(3, 32, tanh),                 # Dense Layer 3 -> 32 with `tanh` activation
            Dense(32, 2, tanh),                 # Dense Layer 32 -> 2 with `tanh` activation 
            postProcess,                        # post process `dx`
            dx -> cacheRetrieve(5:6, dx),       # dynamics FMU | dynamics ANN
            gates,                              # compute resulting dx from ANN + FMU
            dx -> cacheRetrieve(1:4, dx))       # stack together: dx[1,2,3,4] from cache + dx[5,6] from gates

    # new NeuralFMU 
    neuralFMU = ME_NeuralFMU(f, net, (tStart, tStop); saveat=tSave)
    neuralFMU.modifiedState = false # speed optimization (NeuralFMU state equals FMU state)
    
    return neuralFMU 
end
```

Let's test the NeuralFMU: First, load the FMU und built a NeuralFMU from it.


```julia
# load our FMU (we take one from the FMIZoo.jl, exported with Dymola 2022x)
fmu = fmiLoad("VLDM", "Dymola", "2020x"; type=:ME)

# build NeuralFMU
neuralFMU = build_NFMU(fmu)
```

Next, do a simulation for a given start state `x0` from *FMIZoo.jl*.


```julia
# get start state vector from data (FMIZoo)
x0 = FMIZoo.getStateVector(data, tStart)

# simulate and plot the (uninitialized) NeuralFMU
resultNFMU = neuralFMU(x0,                          # the start state to solve the ODE
                       (tStart, tStop);             # the simulation range
                       parameters=data.params,      # the parameters for the VLDM
                       showProgress=showProgress)   # show progress (or not)

display(resultNFMU)     
```

Now, let's have a look on the cumulative consumption plot ...


```julia
# plot the NeuralFMU, original FMU and data (cumulative consumption)
fig = plot(resultNFMU; stateIndices=6:6, stateEvents=false, timeEvents=false, label="NeuralFMU (untrained)", ylabel="cumulative consumption [Ws]")
plot!(fig, resultFMU; stateIndices=6:6, values=false, stateEvents=false, timeEvents=false, label="FMU")
plot!(fig, data.cumconsumption_t, data.cumconsumption_val, label="Data")
```

Finally, unload the FMU and invalidate the NeuralFMU.


```julia
# unload FMU / invalidate NeuralFMU
fmiUnload(fmu)
neuralFMU = nothing
```

## 4. Training the NeuralFMU

An untrained NeuralFMU is not that impressive - so let's train it a bit. 


```julia
# prepare training data 
train_t = data.consumption_t 

# data is as "array of arrays" required (multidimensional data)
train_data = collect([d] for d in data.cumconsumption_val);
```

Next, the loss function is defined. The loss is computed on basis of a given `solution` and `data`. Dependent on the hyperparameter `LOSS`, either `:MAE` or `:MSE` is used to compute the loss. The hyperparameter `LASTWEIGHT` determines how much the last solution point is weight against the remaining solution points. For example a value of $0.3$ determines that the last point of the solution contributes $30\%$ to the loss, whereas all remaining solution points contribute $70\%$ in total.


```julia
function _lossFct(solution::FMU2Solution, data::VLDM_Data, LOSS::Symbol, LASTWEIGHT::Real)

    if !solution.success
        @warn "The solution process was not successfull, try to compute loss anyway."
    end

    # determine the indices in the data array (sampled with 10Hz)
    ts = 1+round(Int, solution.states.t[1]/dt)
    te = 1+round(Int, solution.states.t[end]/dt)
    
    nfmu_cumconsumption = fmiGetSolutionState(solution, 6; isIndex=true)
    cumconsumption = data.cumconsumption_val[ts:te]
    cumconsumption_dev = data.cumconsumption_dev[ts:te]

    Δcumconsumption = 0.0
    if LOSS == :MAE
        Δcumconsumption = FMIFlux.Losses.mae_last_element_rel_dev(nfmu_cumconsumption, cumconsumption, cumconsumption_dev, LASTWEIGHT)
    elseif LOSS == :MSE
        Δcumconsumption = FMIFlux.Losses.mse_last_element_rel_dev(nfmu_cumconsumption, cumconsumption, cumconsumption_dev, LASTWEIGHT)
    else
        @assert false, "Unknown LOSS: `$(LOSS)`"
    end
    
    return Δcumconsumption 
end
```

Finally, the function `train!` is defined, that triggers a new training run for a given set of hyperparameters `hyper_params`, a training ressource `ressource` and the current training index `ind`. The following hyperparameters are used:
- `TRAINDUR` equals the given `ressource` and specifies the training duration (meassured on data) in seconds. Hyperparameter optimizers like *Hyperband* use different ressources during searching for a minimum.
- `ETA` the update rate $\eta$ of the *Adam* optimizer.
- `BETA1` the first momentum coefficient $\beta_1$ of the *Adam* optimizer.
- `BETA2` the second momentum coefficient $\beta_2$ of the *Adam* optimizer. 
- `BATCHDUR` the duration of a single batch element (length) in seconds.
- `LASTWEIGHT` a weighting factor between the last solution point and all remaining solution points. 
- `SCHEDULER` an identifier for the batch scheduler, can be `:Sequential`, `:Random` or `:LossAccumulation`.
- `LOSS` an identifier for the loss function to use, `:MAE` or `:MSE`.


```julia
# ressource = training time horizon (duration of data seen)
function train!(hyper_params, ressource, ind)

    # make the runs determinisitic by fixing the random seed
    Random.seed!(1234)

    # training duration (in seconds) equals the given ressource
    TRAINDUR = ressource

    # unpack the hyperparemters
    ETA, BETA1, BETA2, BATCHDUR, LASTWEIGHT, SCHEDULER, LOSS = hyper_params

    # compute the number of training steps TRAINDUR / BATCHDUR, but do at least one step
    steps = max(round(Int, TRAINDUR/BATCHDUR), 1) 

    # print a bit of info
    @info "--------------\nStarting run $(ind) with parameters: $(hyper_params) and ressource $(ressource) doing $(steps) step(s).\n--------------------"

    # load our FMU (we take one from the FMIZoo.jl, exported with Dymola 2020x)
    fmu = fmiLoad("VLDM", "Dymola", "2020x"; type=:ME) 

    # built the NeuralFMU on basis of the loaded FMU `fmu`
    neuralFMU = build_NFMU(fmu)

    # switch to a more efficient execution configuration, allocate only a single FMU instance, see:
    # https://thummeto.github.io/FMI.jl/dev/features/#Execution-Configuration
    fmu.executionConfig = FMI.FMIImport.FMU2_EXECUTION_CONFIGURATION_NOTHING
    c, _ = FMIFlux.prepareSolveFMU(neuralFMU.fmu, nothing, neuralFMU.fmu.type, true, false, false, false, true, data.params; x0=x0)

    # batch the data (time, targets), train only on model output index 6, plot batch elements
    batch = batchDataSolution(neuralFMU,                            # our NeuralFMU model
                              t -> FMIZoo.getStateVector(data, t),  # a function returning a start state for a given time point `t`, to determine start states for batch elements
                              train_t,                              # data time points
                              train_data;                           # data cumulative consumption 
                              batchDuration=BATCHDUR,               # duration of one batch element
                              indicesModel=6:6,                     # model indices to train on (6 equals the state `cumulative consumption`)
                              plot=false,                           # don't show intermediate plots (try this outside of Jupyter)
                              parameters=data.params,               # use the parameters (map file paths) from *FMIZoo.jl*
                              showProgress=showProgress)            # show or don't show progess bar, as specified at the very beginning

    # limit the maximum number of solver steps to 1000 * BATCHDUR (longer batch elements get more steps)
    # this allows the NeuralFMU to do 10x more steps (average) than the original FMU, but more should not be tolerated (to stiff system)
    solverKwargsTrain = Dict{Symbol, Any}(:maxiters => round(Int, 1000*BATCHDUR)) 
    
    # a smaller dispatch for our custom loss function, only taking the solution object
    lossFct = (solution::FMU2Solution) -> _lossFct(solution, data, LOSS, LASTWEIGHT)

    # selecting a scheduler for training
    scheduler = nothing
    if SCHEDULER == :Random
        # a scheduler that picks a random batch element
        scheduler = RandomScheduler(neuralFMU, batch; applyStep=1, plotStep=0)
    elseif SCHEDULER == :Sequential
        # a scheduler that picks one batch element after another (in chronological order)
        scheduler = SequentialScheduler(neuralFMU, batch; applyStep=1, plotStep=0)
    elseif SCHEDULER == :LossAccumulation
        # a scheduler that picks the element with largest accumulated loss:
        # - after every training step, the accumulated loss for every batch element is increased by the current loss value 
        # - when picking a batch element, the accumulated loss is reset to zero
        # - this promotes selecting elements with larger losses more often, but also prevents starving of elements with small losses
        scheduler = LossAccumulationScheduler(neuralFMU, batch, lossFct; applyStep=1, plotStep=0, updateStep=1)
    else 
        @error "Unknown SCHEDULER: ´$(SCHEDULER)´."
        return nothing
    end

    # loss for training, do a simulation run on a batch element taken from the scheduler
    loss = p -> FMIFlux.Losses.loss(neuralFMU,                          # the NeuralFMU to simulate
                                    batch;                              # the batch to take an element from
                                    p=p,                                # the NeuralFMU training parameters (given as input)
                                    parameters=data.params,             # the FMU paraemters
                                    lossFct=lossFct,                    # our custom loss function
                                    batchIndex=scheduler.elementIndex,  # the index of the batch element to take, determined by the choosen scheduler
                                    logLoss=true,                       # log losses after every evaluation
                                    showProgress=showProgress,          # sho progress bar (or don't)
                                    solverKwargsTrain...)               # the solver kwargs defined above

    # gather the parameters from the NeuralFMU
    params = FMIFlux.params(neuralFMU)

    # initialize the scheduler 
    initialize!(scheduler; parameters=data.params, p=params[1], showProgress=showProgress)
    
    # initialize Adam optimizer with our hyperparameters
    optim = Adam(ETA, (BETA1, BETA2))
   
    # the actual training
    FMIFlux.train!(loss,                            # the loss function for training
                   params,                          # the parameters to train
                   Iterators.repeated((), steps),   # an iterator repeating `steps` times
                   optim;                           # the optimizer to train
                   gradient=:ForwardDiff,           # we use *ForwardDiff.jl* for gradient determination
                   chunk_size=32,                   # the ForwardDiff chunk size is 32, so 32 parameter sensitivites are determined at a time
                   cb=() -> update!(scheduler),     # update the scheduler after every step 
                   proceed_on_assert=true)          # proceed, even if assertions are thrown, with the next step
    
    # switch back to the default execution configuration, allocate a new FMU instance for every run, see:
    # https://thummeto.github.io/FMI.jl/dev/features/#Execution-Configuration
    fmu.executionConfig = FMI.FMIImport.FMU2_EXECUTION_CONFIGURATION_NO_RESET
    FMIFlux.finishSolveFMU(neuralFMU.fmu, c, false, true)

    # save our result parameters
    fmiSaveParameters(neuralFMU, joinpath(@__DIR__, "params", "$(HPRUN)", "$(ind).jld2"))
    
    # simulate the NeuralFMU on a validation trajectory
    resultNFMU = neuralFMU(x0, (data_validation.consumption_t[1], data_validation.consumption_t[end]); parameters=data_validation.params, showProgress=showProgress, maxiters=1e7, saveat=data_validation.consumption_t)

    # determine loss on validation data (if the simulation was successfull)
    validation_loss = nothing 
    if resultNFMU.success
        validation_loss = _lossFct(resultNFMU, data_validation, :MSE)
    end        

    # unload FMU
    fmiUnload(fmu)

    # return the loss (or `nothing` if no loss can be determined)
    return validation_loss
end
```

Let's check if the train function is working for a given set of hyperparameters.


```julia
# check if the train function is working for a set of given hyperparameters
#     ([  ETA, BETA1,  BETA2, BATCHDUR, LASTWEIGHT,         SCHEDULER, LOSS], RESSOURCE, INDEX)
train!([0.001,   0.9, 0.9999,     10.0,        0.7, :LossAccumulation, :MSE],      10.0,     1) 
```

# 5. Results Discussion
After hyperparameter optimization, results can be loaded (one is already preparred if you skipped the optimization).


```julia
# load our FMU (we take one from the FMIZoo.jl, exported with Dymola 2022x)
fmu = fmiLoad("VLDM", "Dymola", "2020x"; type=:ME)

# build NeuralFMU
neuralFMU = build_NFMU(fmu)

# load parameters from hyperparameter optimization
fmiLoadParameters(neuralFMU, joinpath(@__DIR__, "juliacon_2023.jld2"))

# simulate and plot the NeuralFMU
resultNFMU = neuralFMU(x0, (tStart, tStop); parameters=data.params, showProgress=showProgress, saveat=tSave) 
resultFMU = fmiSimulate(fmu, (tStart, tStop); parameters=data.params, showProgress=showProgress, saveat=tSave) 

# plot the NeuralFMU, original FMU and data (cumulative consumption)
fig = plot(resultNFMU; stateIndices=6:6, stateEvents=false, timeEvents=false, label="NeuralFMU", ylabel="cumulative consumption [m/s]")
plot!(fig, resultFMU; stateIndices=6:6, values=false, stateEvents=false, timeEvents=false, label="FMU")
plot!(fig, data.cumconsumption_t, data.cumconsumption_val, label="Data")
```

We also have a ready-to-use function that calculates different errors.


```julia
plotCumulativeConsumption(resultNFMU, resultFMU, data; filename=joinpath(@__DIR__, "comparision_train_100.png"))
```

Because the deviation is small, let's check the last 10% of WLTC focussed.


```julia
plotCumulativeConsumption(resultNFMU, resultFMU, data; range=(0.9, 1.0), filename=joinpath(@__DIR__, "comparision_train_10.png"))
```

Finally, we should check the results on validation data: The full WLTC cycle.


```julia
# get start and stop for the validation cycle (full WLTC)
tStart_validation = data_validation.cumconsumption_t[1]
tStop_validation = data_validation.cumconsumption_t[end]
tSave_validation = data_validation.cumconsumption_t

# simulate the NeuralFMU on validation data
resultNFMU = neuralFMU(x0, (tStart_validation, tStop_validation); parameters=data_validation.params, showProgress=showProgress, saveat=tSave_validation) 
resultFMU = fmiSimulate(fmu, (tStart_validation, tStop_validation); parameters=data_validation.params, showProgress=showProgress, saveat=tSave_validation) 

plotCumulativeConsumption(resultNFMU, resultFMU, data_validation; filename=joinpath(@__DIR__, "comparision_validation_100.png"))
```

... and the last 10% ...


```julia
plotCumulativeConsumption(resultNFMU, resultFMU, data_validation; range=(0.9, 1.0), filename=joinpath(@__DIR__, "comparision_validation_10.png"))
```

After we finished, let's finally unload the FMU and invalidate the NeuralFMU.


```julia
# unload FMU / invalidate NeuralFMU
fmiUnload(fmu)
neuralFMU = nothing
```

# Optional: Organize as module
If you want, you can place all code inside of a module named `NODE_Training`, this simplifies hyper parameter optimization (if you want to do one).


```julia
# for hyper parameter optimization, place the code in a `module`
# just uncomment the following line (and the one at the very beginning, too!) 
#end # NODE_Training 
```
