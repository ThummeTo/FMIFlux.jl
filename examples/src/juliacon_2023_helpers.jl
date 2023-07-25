# Copyright (c) 2023 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. 
# See LICENSE (https://github.com/thummeto/FMIFlux.jl/blob/main/LICENSE) file in the project root for details.

using LaTeXStrings

import FMIFlux: roundToLength
import FMIZoo: movavg

import FMI: FMU2Solution
import FMIZoo: VLDM, VLDM_Data

function fmiSingleInstanceMode(fmu::FMU2, mode::Bool)
    if mode 
        # switch to a more efficient execution configuration, allocate only a single FMU instance, see:
        # https://thummeto.github.io/FMI.jl/dev/features/#Execution-Configuration
        fmu.executionConfig = FMI.FMIImport.FMU2_EXECUTION_CONFIGURATION_NOTHING
        c, _ = FMIFlux.prepareSolveFMU(fmu, nothing, fmu.type, true, false, false, false, true, data.params; x0=x0)
    else
        c = FMI.getCurrentComponent(fmu)
        # switch back to the default execution configuration, allocate a new FMU instance for every run, see:
        # https://thummeto.github.io/FMI.jl/dev/features/#Execution-Configuration
        fmu.executionConfig = FMI.FMIImport.FMU2_EXECUTION_CONFIGURATION_NO_RESET
        FMIFlux.finishSolveFMU(fmu, c, false, true)
    end
    return nothing
end

function dataIndexForTime(t::Real)
    return 1+round(Int, t/dt)
end

function plotEnhancements(neuralFMU::NeuralFMU, fmu::FMU2, data::FMIZoo.VLDM_Data; reductionFactor::Int=10, mov_avg::Int=100, filename=nothing)
    colorMin = 0
    colorMax = 0
    okregion = 0
    label=""

    tStart = data.consumption_t[1]
    tStop = data.consumption_t[end]
    x0 = FMIZoo.getStateVector(data, tStart)
    resultNFMU = neuralFMU(x0, (tStart, tStop); parameters=data.params, showProgress=false, recordValues=:derivatives, saveat=data.consumption_t)
    resultFMU = fmiSimulate(fmu, (tStart, tStop); parameters=data.params, showProgress=false, recordValues=:derivatives, saveat=data.consumption_t) 

    # Finite differences for acceleration
    dt = data.consumption_t[2]-data.consumption_t[1]
    acceleration_val = (data.speed_val[2:end] - data.speed_val[1:end-1]) / dt
    acceleration_val = [acceleration_val..., 0.0]
    acceleration_dev = (data.speed_dev[2:end] - data.speed_dev[1:end-1]) / dt
    acceleration_dev = [acceleration_dev..., 0.0]

    ANNInputs = fmiGetSolutionValue(resultNFMU, :derivatives) # collect([0.0, 0.0, 0.0, data.speed_val[i], acceleration_val[i], data.consumption_val[i]] for i in 1:length(data.consumption_t))
    ANNInputs = collect([ANNInputs[1][i], ANNInputs[2][i], ANNInputs[3][i], ANNInputs[4][i], ANNInputs[5][i], ANNInputs[6][i]] for i in 1:length(ANNInputs[1]))
    
    ANNOutputs = fmiGetSolutionDerivative(resultNFMU, 5:6; isIndex=true)
    ANNOutputs = collect([ANNOutputs[1][i], ANNOutputs[2][i]] for i in 1:length(ANNOutputs[1]))

    FMUOutputs = fmiGetSolutionDerivative(resultFMU, 5:6; isIndex=true)
    FMUOutputs = collect([FMUOutputs[1][i], FMUOutputs[2][i]] for i in 1:length(FMUOutputs[1]))

    ANN_consumption = collect(o[2] for o in ANNOutputs) 
    ANN_error = ANN_consumption - data.consumption_val
    ANN_error = collect(ANN_error[i] > 0.0 ? max(0.0, ANN_error[i]-data.consumption_dev[i]) : min(0.0, ANN_error[i]+data.consumption_dev[i]) for i in 1:length(data.consumption_t))

    FMU_consumption = collect(o[2] for o in FMUOutputs) 
    FMU_error = FMU_consumption - data.consumption_val
    FMU_error = collect(FMU_error[i] > 0.0 ? max(0.0, FMU_error[i]-data.consumption_dev[i]) : min(0.0, FMU_error[i]+data.consumption_dev[i]) for i in 1:length(data.consumption_t))
    
    colorMin=-231.0
    colorMax=231.0
    
    FMU_error = movavg(FMU_error, mov_avg)
    ANN_error = movavg(ANN_error, mov_avg)
    
    ANN_error = ANN_error .- FMU_error

    ANNInput_vel = collect(o[4] for o in ANNInputs)
    ANNInput_acc = collect(o[5] for o in ANNInputs)
    ANNInput_con = collect(o[6] for o in ANNInputs)
    
    _max = max(ANN_error...)
    _min = min(ANN_error...)
    neutral = 0.5 

    if _max > colorMax
        @warn "max value ($(_max)) is larger than colorMax ($(colorMax)) - values will be cut"
    end

    if _min < colorMin
        @warn "min value ($(_min)) is smaller than colorMin ($(colorMin)) - values will be cut"
    end

    anim = @animate for ang in 0:5:360
        l = Plots.@layout [Plots.grid(3,1) r{0.85w}]
        fig = Plots.plot(layout=l, size=(1600,800), left_margin = 10Plots.mm, right_margin = 10Plots.mm, bottom_margin = 10Plots.mm)
    
        colorgrad = cgrad([:green, :white, :red], [0.0, 0.5, 1.0]) # , scale = :log)
    
        scatter!(fig[1], ANNInput_vel[1:reductionFactor:end], ANNInput_acc[1:reductionFactor:end],
                    xlabel="velocity [m/s]", ylabel="acceleration [m/s^2]",
                    color=colorgrad, zcolor=ANN_error[1:reductionFactor:end], label=:none, colorbar=:none) # 
    
        scatter!(fig[2], ANNInput_acc[1:reductionFactor:end], ANNInput_con[1:reductionFactor:end],
                    xlabel="acceleration [m/s^2]", ylabel="consumption [W]",
                    color=colorgrad, zcolor=ANN_error[1:reductionFactor:end], label=:none, colorbar=:none) # 
                
        scatter!(fig[3], ANNInput_vel[1:reductionFactor:end], ANNInput_con[1:reductionFactor:end],
                    xlabel="velocity [m/s]", ylabel="consumption [W]",
                    color=colorgrad, zcolor=ANN_error[1:reductionFactor:end], label=:none, colorbar=:none) # 
    
        scatter!(fig[4], ANNInput_vel[1:reductionFactor:end], ANNInput_acc[1:reductionFactor:end], ANNInput_con[1:reductionFactor:end],
                    xlabel="velocity [m/s]", ylabel="acceleration [m/s^2]", zlabel="consumption [W]",
                    color=colorgrad, zcolor=ANN_error[1:reductionFactor:end], markersize=8, label=:none, camera=(ang,20), colorbar_title=" \n\n\n\n" * L"ΔMAE" * " (smoothed)") 
    
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

    if !isnothing(filename)
        return gif(anim, filename; fps=10)
    else
        return gif(anim; fps=10)
    end
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