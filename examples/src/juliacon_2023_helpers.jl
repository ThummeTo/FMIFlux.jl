# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. 
# See LICENSE (https://github.com/thummeto/FMIFlux.jl/blob/main/LICENSE) file in the project root for details.

using LaTeXStrings

import FMIFlux: roundToLength
import FMIZoo:movavg

import FMI: FMU2Solution
import FMIZoo: VLDM, VLDM_Data

function plotANNError(neuralFMU::NeuralFMU, data::FMIZoo.VLDM_Data; reductionFactor::Int=10, field=:consumption, mov_avg::Int=100, filename=nothing)
    colorMin = 0
    colorMax = 0
    okregion = 0
    label=""

    tStart = data.consumption_t[1]
    tStop = data.consumption_t[end]
    x0 = FMIZoo.getStateVector(data, tStart)
    result = neuralFMU(x0, (tStart, tStop); parameters=data.params, showProgress=true, recordValues=:derivatives) 

    # Finite differences for acceleration
    dt = data.consumption_t[2]-data.consumption_t[1]
    acceleration_val = (data.speed_val[2:end] - data.speed_val[1:end-1]) / dt
    acceleration_val = [acceleration_val..., 0.0]
    acceleration_dev = (data.speed_dev[2:end] - data.speed_dev[1:end-1]) / dt
    acceleration_dev = [acceleration_dev..., 0.0]

    ANNInputs = fmiGetSolutionValue(result, :derivatives) # collect([0.0, 0.0, 0.0, data.speed_val[i], acceleration_val[i], data.consumption_val[i]] for i in 1:length(data.consumption_t))
    ANNInputs = collect([ANNInputs[1][i], ANNInputs[2][i], ANNInputs[3][i], ANNInputs[4][i], ANNInputs[5][i], ANNInputs[6][i]] for i in 1:length(ANNInputs[1]))
    ANNOutputs = fmiGetSolutionDerivative(result, 5:6; isIndex=true)
    ANNOutputs = collect([ANNOutputs[1][i], ANNOutputs[2][i]] for i in 1:length(ANNOutputs[1]))

    ANN_error = nothing 

    if field == :consumption
        ANN_consumption = collect(o[2] for o in ANNOutputs)
        ANN_error = ANN_consumption - data.consumption_val
        ANN_error = collect(ANN_error[i] > 0.0 ? max(0.0, ANN_error[i]-data.consumption_dev[i]) : min(0.0, ANN_error[i]+data.consumption_dev[i]) for i in 1:length(data.consumption_t))
        
        label = L"consumption [W]"
        colorMin=-610.0
        colorMax=610.0
    else # :acceleration 
        ANN_acceleration = collect(o[1] for o in ANNOutputs)
        ANN_error = ANN_acceleration - acceleration_val
        ANN_error = collect(ANN_error[i] > 0.0 ? max(0.0, ANN_error[i]-acceleration_dev[i]) : min(0.0, ANN_error[i]+acceleration_dev[i]) for i in 1:length(data.consumption_t))

        label = L"acceleration [m/s^2]"
        colorMin=-0.04
        colorMax=0.04
    end

    ANN_error = movavg(ANN_error, mov_avg)

    ANNInput_vel = collect(o[4] for o in ANNInputs)
    ANNInput_acc = collect(o[5] for o in ANNInputs)
    ANNInput_con = collect(o[6] for o in ANNInputs)
    
    _max = max(ANN_error...)
    _min = min(ANN_error...)
    neutral = -colorMin/(colorMax-colorMin) # -_min/(_max-_min)

    if _max > colorMax
        @warn "max value ($(_max)) is larger than colorMax ($(colorMax)) - values will be cut"
    end

    if _min < colorMin
        @warn "min value ($(_min)) is smaller than colorMin ($(colorMin)) - values will be cut"
    end

    ANN_error = collect(min(max(e, colorMin), colorMax) for e in ANN_error)

    @info "$(_min) $(_max) $(neutral)"

    anim = @animate for ang in 0:5:360
        l = Plots.@layout [Plots.grid(3,1) r{0.85w}]
        fig = Plots.plot(layout=l, size=(1600,800), left_margin = 10Plots.mm, right_margin = 10Plots.mm, bottom_margin = 10Plots.mm)

        colorgrad = cgrad([:orange, :white, :blue], [0.0, 0.5, 1.0]) # , scale = :log)

        scatter!(fig[1], ANNInput_vel[1:reductionFactor:end], ANNInput_acc[1:reductionFactor:end],
                    xlabel=L"velocity [m/s]", ylabel=L"acceleration [m/s^2]",
                    color=colorgrad, zcolor=ANN_error[1:reductionFactor:end], label=:none, colorbar=:none) # 
   
        scatter!(fig[2], ANNInput_acc[1:reductionFactor:end], ANNInput_con[1:reductionFactor:end],
                    xlabel=L"acceleration [m/s^2]", ylabel=L"consumption [W]",
                    color=colorgrad, zcolor=ANN_error[1:reductionFactor:end], label=:none, colorbar=:none) # 
                 
        scatter!(fig[3], ANNInput_vel[1:reductionFactor:end], ANNInput_con[1:reductionFactor:end],
                    xlabel=L"velocity [m/s]", ylabel=L"consumption [W]",
                    color=colorgrad, zcolor=ANN_error[1:reductionFactor:end], label=:none, colorbar=:none) # 

        scatter!(fig[4], ANNInput_vel[1:reductionFactor:end], ANNInput_acc[1:reductionFactor:end], ANNInput_con[1:reductionFactor:end],
                    xlabel=L"velocity [m/s]", ylabel=L"acceleration [m/s^2]", zlabel=L"consumption [W]",
                    color=colorgrad, zcolor=ANN_error[1:reductionFactor:end], markersize=8, label=:none, camera=(ang,20), colorbar_title=" \n\n\n\n" * L"Δ" * label * " (smoothed)") # 
    
        # draw invisible dummys to scale colorbar to fixed size
        for i in 1:3 
            scatter!(fig[i], [0.0,0.0], [0.0,0.0],
                color=colorgrad, zcolor=[colorMin, colorMax], 
                markersize=0, label=:none)
        end
        for i in 4:4
            scatter!(fig[i], [0.0,0.0], [0.0,0.0], [0.0,0.0],
                color=colorgrad, zcolor=[colorMin, colorMax], 
                markersize=0, label=:none)
        end
    end

    return gif(anim, filename; fps=10)
end

function plotCumulativeConsumption(solutionNFMU::FMU2Solution, solutionFMU::FMU2Solution, data::FMIZoo.VLDM_Data; range=(0.0,1.0), filename=nothing)

    len = length(data.consumption_t)
    steps = (1+round(Int, range[1]*len)):(round(Int, range[end]*len))

    t        = data.consumption_t
    nfmu_val = fmiGetSolutionState(solutionNFMU, 6; isIndex=true)
    fmu_val  = fmiGetSolutionState(solutionFMU,  6; isIndex=true)
    data_val = data.cumconsumption_val
    data_dev = data.cumconsumption_dev
    
    mse_nfmu = FMIFlux.Losses.mse_dev(nfmu_val, data_val, data_dev)
    mse_fmu  = FMIFlux.Losses.mse_dev(fmu_val,  data_val, data_dev)
    
    mae_nfmu = FMIFlux.Losses.mae_dev(nfmu_val, data_val, data_dev)
    mae_fmu  = FMIFlux.Losses.mae_dev(fmu_val,  data_val, data_dev)
    
    max_nfmu = FMIFlux.Losses.max_dev(nfmu_val, data_val, data_dev)
    max_fmu  = FMIFlux.Losses.max_dev(fmu_val,  data_val, data_dev)
   
    fig = plot(xlabel=L"t[s]", ylabel=L"x_6 [Ws]", dpi=600)
    plot!(fig, t[steps], data_val[steps]; label="Data", ribbon=data_dev, fillalpha=0.3)
    plot!(fig, t[steps],  fmu_val[steps]; label="FMU            [ MSE:$(roundToLength(mse_fmu,10)) | MAE:$(roundToLength(mae_fmu,10)) | MAX:$(roundToLength(max_fmu,10)) ]")
    plot!(fig, t[steps], nfmu_val[steps]; label="NeuralFMU  [ MSE:$(roundToLength(mse_nfmu,10)) | MAE:$(roundToLength(mae_nfmu,10)) | MAX:$(roundToLength(max_nfmu,10)) ]")

    if !isnothing(filename)
        savefig(fig, filename)
    end

    return fig
end

function simPlotCumulativeConsumption(cycle::Symbol, filename=nothing; kwargs...)
    d = FMIZoo.VLDM(cycle)
    tStart = d.consumption_t[1]
    tStop = d.consumption_t[end]
    tSave = d.consumption_t
    
    resultNFMU = neuralFMU(x0, (tStart, tStop); parameters=d.params, showProgress=false, saveat=tSave, maxiters=1e7) 
    resultFMU = fmiSimulate(fmu, (tStart, tStop), parameters=d.params, showProgress=false, saveat=tSave)

    fig = plotCumulativeConsumption(resultNFMU, resultFMU, d, kwargs...)
    if !isnothing(filename)
        savefig(fig, filename)
    end
    return fig
end

function checkMSE(cycle; init::Bool=false)

    data = FMIZoo.VLDM(cycle)
    tStart = data.consumption_t[1]
    tStop = data.consumption_t[end]
    tSave = data.consumption_t
    
    if init
        c = FMI.FMIImport.getCurrentComponent(fmu)
        FMI.FMIImport.fmi2SetFMUstate(c, batch[1].initialState)
        c.eventInfo = deepcopy(batch[1].initialEventInfo)
        c.t = batch[1].tStart
    end
    resultNFMU = neuralFMU(x0, (tStart, tStop); parameters=data.params, showProgress=true, maxiters=1e7, saveat=tSave) 
    
    mse_NFMU = FMIFlux.Losses.mse_dev(data.cumconsumption_val, fmiGetSolutionState(resultNFMU, 6; isIndex=true), data.cumconsumption_dev)
    
    return mse_NFMU
end
