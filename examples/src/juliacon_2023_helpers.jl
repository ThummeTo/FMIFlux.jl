# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. 
# See LICENSE (https://github.com/thummeto/FMIFlux.jl/blob/main/LICENSE) file in the project root for details.

# Hyperparameters are the results of hyperparameter optimization
ETA, BETA1, BETA2, BATCHDUR, LASTWEIGHT, SCHEDULER, LOSS, EIGENLOSS, STIFFNESSRATIO = [0.001, 0.999, 0.999, 9.0, 0.8, :LossAccumulation, :MSE, :OFF, 0.5]
RESSOURCE = 64.85555555555555
TRAINSTEPS = max(round(Int, RESSOURCE/BATCHDUR), 1) 
MINIMUM =  16.575945029410605

import FMIFlux: roundToLength
import FMIImport: getCurrentComponent, fmi2SetFMUstate
global scaleFac
function checkLoss(init::Bool=false; cycle=:train)

    data = FMIZoo.VLDM(cycle)
    tStart = data.consumption_t[1]
    tStop = data.consumption_t[end]
    tSave = data.consumption_t
    data.params["drivingCycle.combiTimeTable.timeEvents"] = 1
    data.params["drivingCycle.combiTimeTable1.timeEvents"] = 1
    
    if init
        c = getCurrentComponent(fmu)
        fmi2SetFMUstate(c, batch[1].initialState)
        c.eventInfo = deepcopy(batch[1].initialEventInfo)
        c.t = batch[1].tStart
    end
    cl_resultNFMU = neuralFMU(x0, (tStart, tStop); parameters=data.params, showProgress=true, maxiters=1e7, saveat=tSave, recordEigenvalues=true) # [120s]
    
    if init
        c = getCurrentComponent(fmu)
        fmi2SetFMUstate(c, batch[1].initialState)
        c.eventInfo = deepcopy(batch[1].initialEventInfo)
        c.t = batch[1].tStart
    end
    cl_resultFMU = fmiSimulate(fmu, (tStart, tStop), parameters=data.params, recordValues=:derivatives, saveat=tSave, recordEigenvalues=true)

    mse_NFMU = FMIFlux.Losses.mse(data.cumconsumption_val, fmiGetSolutionState(cl_resultNFMU, 6; isIndex=true))
    max_NFMU = max(abs.(data.cumconsumption_val .- fmiGetSolutionState(cl_resultNFMU, 6; isIndex=true))...)

    mse_FMU  = FMIFlux.Losses.mse(data.cumconsumption_val, fmiGetSolutionState(cl_resultFMU, 6; isIndex=true))
    max_FMU  = max(abs.(data.cumconsumption_val .- fmiGetSolutionState(cl_resultFMU, 6; isIndex=true))...)

    global scaleFac 
    if cycle == :train || cycle == "WLTCC2_Low"
        d1 = data.cumconsumption_val
        d2 = fmiGetSolutionState(cl_resultFMU, 6; isIndex=true)
        dev = data.cumconsumption_dev 
        opt = Optim.optimize(p -> objective(p, d1, d2, dev), [0.95]; iterations=250) # do max. 250 iterations
        mse_scale = opt.minimum
        #scaleFac = opt.minimizer[1]
        scaleFac = d1[end]/d2[end]
        @info "New computed scaling factor for cycle `$(cycle)` is $(round(scaleFac, digits=6)) with loss $(mse_scale)"
    end
    mse_scale = FMIFlux.Losses.mse(data.cumconsumption_val, fmiGetSolutionState(cl_resultFMU, 6; isIndex=true) .* scaleFac)
    max_scale = max(abs.(data.cumconsumption_val .- fmiGetSolutionState(cl_resultFMU, 6; isIndex=true) .* scaleFac)...)

    println("Errors [$(cycle)]   | better FMU? | better FMU (scaled)?")
    println("  NeuralFMU | MSE = $(roundToLength(mse_NFMU , 12))   ($(mse_NFMU < mse_FMU ? '+' : '-')$(round(abs((mse_FMU/mse_NFMU-1)*100);digits=1))%)   ($(mse_NFMU < mse_scale ? '+' : '-')$(round(abs((mse_scale/mse_NFMU-1)*100);digits=1))%)")
    println("            | MAX = $(roundToLength(max_NFMU , 12))   ($(max_NFMU < max_FMU ? '+' : '-')$(round(abs((max_FMU/max_NFMU-1)*100);digits=1))%)   ($(max_NFMU < max_scale ? '+' : '-')$(round(abs((max_scale/max_NFMU-1)*100);digits=1))%)")
    println("  FMU       | MSE = $(roundToLength(mse_FMU  , 12))")
    println("            | MAX = $(roundToLength(max_FMU  , 12))")
    println("  FMU (sca) | MSE = $(roundToLength(mse_scale, 12))")
    println("            | MAX = $(roundToLength(max_scale, 12))")

    fig = fmiPlot(cl_resultNFMU; title="$(cycle)", stateIndices=6:6, values=false, stateEvents=false, label="NeuralFMU ($(roundToLength(mse_NFMU, 12)))");
    fmiPlot!(fig, cl_resultFMU; stateIndices=6:6, stateEvents=false, values=false, label="FMU ($(roundToLength(mse_FMU, 12)))");
    Plots.plot!(fig, data.consumption_t, fmiGetSolutionState(cl_resultFMU, 6; isIndex=true) .* scaleFac, label="FMU (scaled) ($(roundToLength(mse_scale, 12)))");
    Plots.plot!(fig, data.consumption_t, data.cumconsumption_val, label="Data", ribbon=data.cumconsumption_dev, fillalpha=0.3)
    display(fig) 

    max_re = -Inf 
    min_re = Inf
    for i in 2:length(cl_resultNFMU.eigenvalues.t)
        eigvals = cl_resultNFMU.eigenvalues.saveval[i]

        for j in 1:Int(length(eigvals)/2)
            re = eigvals[(j-1)*2+1]
            im = eigvals[j*2]

            if re > max_re 
                max_re = re
            end

            if re < min_re 
                min_re = re 
            end
        end
    end

    @info "Eigenvalues between real $(min_re) and $(max_re)"
    @info "Events: $(length(cl_resultNFMU.events)) VS $(length(cl_resultFMU.events))"

    return mse_NFMU
end

function validate(init::Bool=false)
    cycles = [:train, :test, :validate]

    for cycle in cycles
        checkLoss(init; cycle=cycle)
    end
end