### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 89a683f9-4e69-4daf-80bd-486f3a135797
# [TODO] remove this block!
let
    import Pkg
    Pkg.activate("C:\\Users\\thummeto\\.julia\\environments\\v1.10")
	using Revise
end

# ╔═╡ a1ee798d-c57b-4cc3-9e19-fb607f3e1e43
using PlutoUI # Notebook UI

# ╔═╡ 72604eef-5951-4934-844d-d2eb7eb0292c
using FMI # load and simulate FMUs

# ╔═╡ 21104cd1-9fe8-45db-9c21-b733258ff155
using FMIFlux # machine learning with FMUs

# ╔═╡ eaae989a-c9d2-48ca-9ef8-fd0dbff7bcca
using FMIFlux.Flux # default Julia Machine Learning library

# ╔═╡ 98c608d9-c60e-4eb6-b611-69d2ae7054c9
using FMIFlux.DifferentialEquations # [TODO]

# ╔═╡ 9d9e5139-d27e-48c8-a62e-33b2ae5b0086
using FMIZoo # a collection of demo FMUs

# ╔═╡ 45c4b9dd-0b04-43ae-a715-cd120c571424
using Plots, PlotlyBase, PlotlyKaleido

# ╔═╡ 1470df0f-40e1-45d5-a4cc-519cc3b28fb8
md"""
# Hands-on:$br Hybrid Modeling using FMI

Workshop @ MODPROD 2024 (Linköping University, Sweden)

by Tobias Thummerer, Lars Mikelsons (University of Augsburg)

*#hybridmodeling, #sciml, #neuralode, #neuralfmu, #penode*

# Abstract
If there is something YOU know about a physical system, AI shouldn’t need to learn it. How to integrate YOUR system knowledge into a ML development process is the core topic of this hands-on workshop. The entire workshop evolves around a challenging use case from robotics: Modeling a robot that is able to write arbitrary messages with a pen. After introducing the topic and the considered use case, participants can experiment with their very own hybrid model topology. 

# Introduction
This workshop focuses on the integration of Functional Mock-Up Units (FMUs) into a machine learning topology. FMUs are simulation models that can be generated within a variety of modeling tools, see the [FMI homepage](https://fmi-standard.org/). Together with deep neural networks that complement and improve the FMU prediction, so called *NeuralFMUs* can be created. 
The workshop itself evolves around the hybrid modeling of a *Selective Compliance Assembly Robot Arm* (SCARA), that is able to write user defined words on a sheet of paper. A ready to use physical simulation model (FMU) for the SCARA is given and shortly highlighted in this workshop. However, this model – as any simulation model – shows some deviations if compared to measurements from the real system. These deviations results from unmodeled slip-stick-friction: The pen sticks to the paper until a force limit is reached, but then moves jerkily. A hard to model physical effect – but not for a NeuralFMU.

## Example Video
If you haven't seen such a system yet, you can watch the following video. There are many more similar videos out there.
"""

# ╔═╡ 7d694be0-cd3f-46ae-96a3-49d07d7cf65a
html"""
<iframe width="560" height="315" src="https://www.youtube.com/embed/ryIwLLr6yRA?si=ncr1IXlnuNhWPWgl" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
"""

# ╔═╡ 6fc16c34-c0c8-48ce-87b3-011a9a0f4e7c
md"""
This video is by *Alexandru Babaian* on YouTube.

## Requirements
To follow this workshop, you should ...
- ... have a rough idea what the *Functional Mock-Up Interface* is and how the standard-conform models - the *Functional Mock-Up Units* - work. If not, a good source is the homeapge of the standard, see the [FMI Homepage](https://fmi-standard.org/).
- ... know the *Julia Programming Language* or at least have some programming skills in another high-level programming language like *Python* or *Matlab*. An introduction to Julia can be found on the [Julia Homepage](https://julialang.org/), but there are many more introductions in different formats available on the internet.
- ... have an idea of how modeling (in terms of modeling ODE and DAE systems) and simulation (solving) of such models works.

The technical requirements are:

|   | recommended | minimum |
| ----- | ---- | ---- |
| RAM | >= 16GB | 8GB |
| OS | Windows | Windows / Linux |
| Julia | 1.10 | 1.6 |
"""

# ╔═╡ 8a82d8c7-b781-4600-8780-0a0a003b676c
md"""
# Loading required Julia libraries
Before starting with the actual coding, we load in the required Julia libraries. 
This Pluto-Notebooks installs all required packages automatically.
However, this will take some minutes when you start the notebook for the first time...
"""

# ╔═╡ 5cb505f7-01bd-4824-8876-3e0f5a922fb7
md""" 
Load in the plotting library ...
"""

# ╔═╡ 1e9541b8-5394-418d-8c27-2831951c538d
md"""
... and use the beutiful `plotly` backend for interactive plots.
"""

# ╔═╡ bc077fdd-069b-41b0-b685-f9513d283346
plotly()

# ╔═╡ 7b82e333-e133-4b7b-8951-5a8b6a0155db
md"""
The following line just adds a table of contents to the notebook sidebar.
"""

# ╔═╡ 1bb3d6ae-af9c-4b5d-b2e7-19cee7750347
TableOfContents()

# ╔═╡ 44500f0a-1b89-44af-b135-39ce0fec5810
md"""
Next, we define some helper functions, that are not important to follow the workshop - they are hidden by default. However they are here, if you want to explore what it takes to write fully working code. If you do this workshop for the first time, it is recommended to skip the hidden part and directly go on with chapter 1.
"""

# ╔═╡ 915e4601-12cc-4b7e-b2fe-574e116f3a92
md"""
# Loading Model (FMU) and Data
We want to do hybrid modeling, so we need a simulation model and some data to work with. Fortuneately, someone already prepared both for us. We start by loading some data from the FMIZoo, which is a collection of FMUs and corresponding data.
"""

# ╔═╡ f8e40baa-c1c5-424a-9780-718a42fd2b67
md"""
## Training Data
"""

# ╔═╡ 74289e0b-1292-41eb-b13b-a4a5763c72b0
# load training data for the `RobotRR` from the FMIZoo
data_train = FMIZoo.RobotRR(:train)

# ╔═╡ 33223393-bfb9-4e9a-8ea6-a3ab6e2f22aa
begin

LIVE_RESULTS_MESSAGE = "ℹ️ Live Results are disabled to safe performance. Checkbox `Enable Live Results`."
LIVE_TRAIN_MESSAGE = "ℹ️ Live Training is disabled to safe performance. Checkbox `Enable Live Training`."
HIDDEN_CODE_MESSAGE = "👻 Hidden Code | You probably want to skip this code section on the first run."

import FMI.FMIImport.FMICore: hasCurrentComponent, getCurrentComponent, FMU2Solution
import Random 
	
function fmiSingleInstanceMode!(fmu::FMU2, 
	mode::Bool, 
	params=FMIZoo.getParameter(data_train, 0.0; friction=false), 
	x0=FMIZoo.getState(data_train, 0.0))

	fmu.executionConfig = deepcopy(FMU2_EXECUTION_CONFIGURATION_NO_RESET)

	# for this model, state events are generated but don't need to be handled,
	# we can skip that to gain performance
    fmu.executionConfig.handleStateEvents = false

    if mode 
        # switch to a more efficient execution configuration, allocate only a single FMU instance, see:
        # https://thummeto.github.io/FMI.jl/dev/features/#Execution-Configuration
        fmu.executionConfig.terminate = true
        fmu.executionConfig.instantiate = false
        fmu.executionConfig.reset = true
        fmu.executionConfig.setup = true
        fmu.executionConfig.freeInstance = false
        c, _ = FMIFlux.prepareSolveFMU(fmu, nothing, fmu.type, true, # instantiate 
                                                    false, # free 
                                                    true, # terminate 
                                                     true, # reset 
                                                     true, # setup
                                                     params; x0=x0)
    else
		if !hasCurrentComponent(fmu)
        	return nothing
		end
		c = getCurrentComponent(fmu)
        # switch back to the default execution configuration, allocate a new FMU instance for every run, see:
        # https://thummeto.github.io/FMI.jl/dev/features/#Execution-Configuration
        fmu.executionConfig.terminate = true 
        fmu.executionConfig.instantiate = true
        fmu.executionConfig.reset = true
        fmu.executionConfig.setup = true
        fmu.executionConfig.freeInstance = true
        FMIFlux.finishSolveFMU(fmu, c, true, # free 
                                            true) # terminate
    end
    return nothing
end

function dividePath(values)
	last_value = values[1]
	paths = []
	path = []
	for j in 1:length(values)
		if values[j] == 1.0
			push!(path, j)
		end

		if values[j] == 0.0 && last_value != 0.0
			push!(path, j)
			push!(paths, path)
			path = []
		end

		last_value = values[j]
	end
	if length(path) > 0
		push!(paths, path)
	end
	return paths
end

function plotRobot(solution::FMU2Solution, t::Real)
	x = solution.states(t)
	a1 = x[5]
	a2 = x[3]

	dt = 0.01
	i = 1+round(Integer, t/dt)
	v = solution.values.saveval[i]

	l1 = 0.2
	l2 = 0.1

	margin = 0.05
	scale = 1500
	fig = plot(; title="Time $(round(t; digits=1))s",
		size=(round(Integer, (2*margin+l1+l2)*scale), round(Integer, (l1+l2+2*margin)*scale)),
		xlims=(-margin, l1+l2+margin), ylims=(-l1-margin, l2+margin), legend=:bottomleft)

	p0 = [0.0, 0.0]
	p1 = p0 .+ [cos(a1)*l1, sin(a1)*l1]
	p2 = p1 .+ [cos(a1+a2)*l2, sin(a1+a2)*l2]

	f_norm = collect(v[3] for v in solution.values.saveval)
	
	paths = dividePath(f_norm)
	drawing = collect(v[1:2] for v in solution.values.saveval)
	for path in paths
		plot!(fig, collect(v[1] for v in drawing[path]), collect(v[2] for v in drawing[path]), label=:none, color=:black, style=:dot)
	end

	paths = dividePath(f_norm[1:i])
	drawing_is = collect(v[4:5] for v in solution.values.saveval)[1:i]
	for path in paths
		plot!(fig, collect(v[1] for v in drawing_is[path]), collect(v[2] for v in drawing_is[path]), label=:none, color=:green, width=2)
	end

	plot!(fig, [p0[1], p1[1]], [p0[2], p1[2]], label=:none, width=3, color=:blue)
	plot!(fig, [p1[1], p2[1]], [p1[2], p2[2]], label=:none, width=3, color=:blue)

	scatter!(fig, [p0[1]], [p0[2]], label="R1 | α1=$(round(a1; digits=3)) rad", color=:red)
	scatter!(fig, [p1[1]], [p1[2]], label="R2 | α2=$(round(a2; digits=3)) rad", color=:purple)

	scatter!(fig, [v[1]], [v[2]], label="TCP | F=$(v[3]) N", color=:orange)
end

HIDDEN_CODE_MESSAGE

end # begin

# ╔═╡ f111e772-a340-4217-9b63-e7715f773b2c
md"""
Let's have a look on the data!
You can use the slider to pick a specific point in time.
"""

# ╔═╡ 92ad1a99-4ad9-4b69-b6f3-84aab49db54f
@bind t_train_plot Slider(0.0:0.1:data_train.t[end], default=data_train.t[1])

# ╔═╡ 909de9f1-2aca-4bf0-ba60-d3418964ba4a
plotRobot(data_train.solution, t_train_plot)

# ╔═╡ d8ca5f66-4f55-48ab-a6c9-a0be662811d9
md"""
Let's extract a start and stop time, as well as saving points for our later solution process:
"""

# ╔═╡ 41b1c7cb-5e3f-4074-a681-36dd2ef94454
tSave = data_train.t

# ╔═╡ 8f45871f-f72a-423f-8101-9ce93e5a885b
tStart = tSave[1]

# ╔═╡ 57c039f7-5b24-4d63-b864-d5f808110b91
tStop = tSave[end]

# ╔═╡ 8d93a1ed-28a9-4a77-9ac2-5564be3729a5
md"""
## Validation Data
"""

# ╔═╡ 4a8de267-1bf4-42c2-8dfe-5bfa21d74b7e
# load validation data for the `RobotRR` from the FMIZoo
data_validation = FMIZoo.RobotRR(:validate)

# ╔═╡ 6a8b98c9-e51a-4f1c-a3ea-cc452b9616b7
md"""
Let's have a look on the data!
You can use the slider to pick a specific point in time.
"""

# ╔═╡ dbde2da3-e3dc-4b78-8f69-554018533d35
@bind t_validate_plot Slider(0.0:0.1:data_validation.t[end], default=data_validation.t[1])

# ╔═╡ d42d0beb-802b-4d30-b5b8-683d76af7c10
plotRobot(data_validation.solution, t_validate_plot)

# ╔═╡ 3756dd37-03e0-41e9-913e-4b4f183d8b81
md"""
## Simulation Model (FMU)
It's called `RobotRR` for `Robot Rotational Rotational`, indicating that this robot consists of two rotational joints, connected by links.
"""

# ╔═╡ 2f83bc62-5a54-472a-87a2-4ddcefd902b6
# load the FMU named `RobotRR` from the FMIZoo
# the FMU was exported from Dymola (version 2023x)
# load the FMU in mode `model-exchange` (ME) 
fmu = fmiLoad("RobotRR", "Dymola", "2023x"; type=:ME) 

# ╔═╡ c228eb10-d694-46aa-b952-01d824879287
begin

using FMI.FMIImport: fmi2StringToValueReference

# declare some model identifiers (inside of the FMU)
STATE_I1 = fmu.modelDescription.stateValueReferences[2]
STATE_I2 = fmu.modelDescription.stateValueReferences[1]
STATE_A1  = fmi2StringToValueReference(fmu, "rRPositionControl_Elasticity.rr1.rotational1.revolute1.phi")
STATE_A2  = fmi2StringToValueReference(fmu,"rRPositionControl_Elasticity.rr1.rotational2.revolute1.phi")
STATE_dA1 = fmi2StringToValueReference(fmu,"rRPositionControl_Elasticity.rr1.rotational1.revolute1.w")
STATE_dA2 = fmi2StringToValueReference(fmu,"rRPositionControl_Elasticity.rr1.rotational2.revolute1.w")

DER_ddA2 = fmu.modelDescription.derivativeValueReferences[4]
DER_ddA1 = fmu.modelDescription.derivativeValueReferences[6]

VAR_TCP_PX = fmi2StringToValueReference(fmu,"rRPositionControl_Elasticity.tCP.p_x")
VAR_TCP_PY = fmi2StringToValueReference(fmu,"rRPositionControl_Elasticity.tCP.p_y")
VAR_TCP_VX = fmi2StringToValueReference(fmu,"rRPositionControl_Elasticity.tCP.v_x")
VAR_TCP_VY = fmi2StringToValueReference(fmu,"rRPositionControl_Elasticity.tCP.v_y")
VAR_TCP_F  = fmi2StringToValueReference(fmu, "combiTimeTable.y[3]")
	
HIDDEN_CODE_MESSAGE
	
end

# ╔═╡ f168b997-355d-4d01-9f18-caf3d77194a5
# [TODO] comment
fmiSingleInstanceMode!(fmu, true)

# ╔═╡ dd0bdcf5-37a4-4004-934a-d535dd86c27e
md"""
[TODO fmiInfo]
We weill start any simulation at `x0`:
"""

# ╔═╡ 276ef0b2-568b-4e85-9b56-88170336d30c
x0 = FMIZoo.getState(data_train, tStart)

# ╔═╡ 634f923a-5e09-42c8-bac0-bf165ab3d12a
solver = Tsit5()

# ╔═╡ 8aed89cf-787d-4f07-a67a-85fe384caa5e
parameters = FMIZoo.getParameter(data_train, tStart; friction=false)

# ╔═╡ 0c9493c4-322e-41a0-9ec7-2e2c54ae1373
recordValues = [DER_ddA2, DER_ddA1, # mechanical accelerations
            STATE_A2, STATE_A1, # mechanical angles
VAR_TCP_PX, VAR_TCP_PY, VAR_TCP_VX, VAR_TCP_VY, VAR_TCP_F]

# ╔═╡ 25e55d1c-388f-469d-99e6-2683c0508693
sol_fmu_train = fmiSimulate(fmu, (tStart, tStop); solver=solver, parameters=parameters, saveat=tSave, recordValues=recordValues)

# ╔═╡ 74c519c9-0eef-4798-acff-b11044bb4bf1
md"""
Now that we know our model and data a little bit better, it's time to care about our hybrid model topology.

# Experiments: $br Hybrid Model Topology

[Todo Picture]
[Todo: allow adding more or less interface signals, show computation time]
"""

# ╔═╡ 786c4652-583d-43e9-a101-e28c0b6f64e4
md"""
## Choosing interface signals
between the physical and machine learning domain

[Todo Text]
[Todo: allow adding more or less interface signals, show computation time]

Choose additional FMU variables to put in together with the state derivatives.
"""

# ╔═╡ b42bf3d8-e70c-485c-89b3-158eb25d8b25
@bind choose_y_refs MultiCheckBox([STATE_A1 => "Angle Joint 1", STATE_A2 => "Angle Joint 2", VAR_TCP_PX => "TCP position x", VAR_TCP_PY => "TCP position y", VAR_TCP_VX => "TCP velocity x", VAR_TCP_VY => "TCP velocity y", VAR_TCP_F => "TCP (normal) force z"])

# ╔═╡ 5d688c3d-b5e3-4a3a-9d91-0896cc001000
md"""
We start building out deep model as a `Chain` of layers. For now, there is only a single layer in it: The FMU `fmu` itself. The layer input `x` is interpreted as system state and set in the fmu call vai `x=x`. Further, we want all state derivatives as layer outputs `dx_refs=:all` and some additional outputs specified via `y_refs=y_refs`. Which signals are use for `y_refs`, can be selected above.
"""

# ╔═╡ 2e08df84-a468-4e99-a277-e2813dfeae5c
model = Chain(x -> fmu(; x=x, dx_refs=:all, y_refs=choose_y_refs))

# ╔═╡ 33791947-342b-4bf4-9d0a-c3979c0ad88a
begin 
	using FMIFlux.FMISensitivity.ReverseDiff
	ben_grad = (_x,) -> ReverseDiff.gradient(x -> sum(model(x)), _x);
	ben_grad(x0)
	@time ben_grad(x0) # second run for "benchmarking"
end

# ╔═╡ 432ab3ff-5336-4413-9807-45d4bf756e1f
begin 
	using FMIFlux.FMISensitivity.ForwardDiff
	ben_fd_grad = (_x,) -> ForwardDiff.gradient(x -> sum(model(x)), _x);
	ben_fd_grad(x0)
	@time ben_fd_grad(x0) # second run for "benchmarking"
end

# ╔═╡ 0a7955e7-7c1a-4396-9613-f8583195c0a8
md"""
Depending on how many signals you select, the output of the FMU-layer is extended. The first six outputs are the state derivatives, the remaining are the additional outputs selected above.
"""

# ╔═╡ 4912d9c9-d68d-4afd-9961-5d8315884f75
model(x0)

# ╔═╡ 6a1a5092-6832-4ac7-9850-346e3fa28252
md"""
Finally, keep in mind that the amount of selected signals has influence on the computational performance of the model. The more signals you use, the slower is inference and gradient determination. The current timing and allocations for inference is:
"""

# ╔═╡ a1aca180-d561-42a3-8d12-88f5a3721aae
begin 
	ben_inf = (_x,) -> model(_x);
	ben_inf(x0);
	@time ben_inf(x0)
end

# ╔═╡ 13ede3cd-99b1-4e65-8a18-9043db544728
md"""
Gradient computation takes a little longer of course. We use reverse-mode Automatic Differentiation via `ReverseDiff.jl` here:
"""

# ╔═╡ 7622ed93-9c66-493b-9fba-c0d3755758a8
md"""
Further, forward-mode Automatic Differentiation is available too via `ForwardDiff.jl`, but a little bit slower than reverse-mode:
"""

# ╔═╡ eaf37128-0377-42b6-aa81-58f0a815276b
md"""
So keep in mind that the choice of interface might have a significant impact on your inference and training performance!
"""

# ╔═╡ c030d85e-af69-49c9-a7c8-e490d4831324
md"""
## Online Data Pre- and Postprocessing
**is required for hybrid models**

Now that we have defined the signals that come *from* the FMU and go *into* the ANN, we need to think about data pre- and post-processing. In ML, this is often done before the actual training starts. In hybrid modeling, we need to do this *online*. This gets more clear if we have a look on the used activation functions, like e.g. the *tanh*.

Let's see what's happening as soon as we put the derivative *angular velocity of joint 1* (dα1) from the FMU into a `tanh` function:
"""

# ╔═╡ 51c200c9-0de3-4e50-8884-49fe06158560
begin 
	fig_pre_post1 = plot(layout=grid(1,2,widths=(1/4, 3/4)), xlabel="t [s]", legend=:bottomright)

	plot!(fig_pre_post1[1], data_train.t, data_train.da1, label=:none, xlims=(0.0,0.1))
	plot!(fig_pre_post1[1], data_train.t, tanh.(data_train.da1), label=:none)
	
	plot!(fig_pre_post1[2], data_train.t, data_train.da1, label="dα1")
	plot!(fig_pre_post1[2], data_train.t, tanh.(data_train.da1), label="tanh(dα1)")
	
	fig_pre_post1
end

# ╔═╡ 0dadd112-3132-4491-9f02-f43cf00aa1f9
md"""
In general, it looks like the velocity isn't saturated too much. This is a good thing and not always the case! However, the very beginning of the trajectory is saturated too much (the peak value of $\approx -3$ is saturated to $\approx -1$). This is bad, because the hybrid model will move *slower* at this point in time and won't reach the same angle as the FMU.

We can add shift and scale operations before and after the ANN to bypass this issue. See how you can influence the output *after* the `tanh` (and the ANN repectively) to match the ranges. 
"""

# ╔═╡ bf6bf640-54bc-44ef-bd4d-b98e934d416e
@bind PRE_POST_SHIFT Slider(-1:0.1:1.0, default=0.0)

# ╔═╡ 5c2308d9-6d04-4b38-af3b-6241da3b6871
md"""
Change the `shift` value $(PRE_POST_SHIFT):
"""

# ╔═╡ 007d6d95-ad85-4804-9651-9ac3703d3b40
@bind PRE_POST_SCALE Slider(0.1:0.1:2.0, default=1.0)

# ╔═╡ 639889b3-b9f2-4a3c-999d-332851768fd7
md"""
Change the `scale` value $(PRE_POST_SCALE):
"""

# ╔═╡ ed1887df-5079-4367-ab04-9d02a1d6f366
begin 
	fun_pre = ShiftScale([PRE_POST_SHIFT], [PRE_POST_SCALE])
	fun_post = ScaleShift(fun_pre)
	
	fig_pre_post2 = plot(;layout=grid(1,2,widths=(1/4, 3/4)), xlabel="t [s]")

	plot!(fig_pre_post2[2], data_train.t, data_train.da1, label=:none, title="Shift: $(round(PRE_POST_SHIFT; digits=1)) | Scale: $(round(PRE_POST_SCALE; digits=1))", legend=:bottomright)
	plot!(fig_pre_post2[2], data_train.t, tanh.(data_train.da1), label=:none)
	plot!(fig_pre_post2[2], data_train.t, fun_post(tanh.(fun_pre(data_train.da1))), label=:none)

	plot!(fig_pre_post2[1], data_train.t, data_train.da1, label="dα1", xlims=(0.0, 0.1))
	plot!(fig_pre_post2[1], data_train.t, tanh.(data_train.da1), label="tanh(dα1)")
	plot!(fig_pre_post2[1], data_train.t, fun_post(tanh.(fun_pre(data_train.da1))), label="post(tanh(pre(dα1)))")
	
	fig_pre_post2
end

# ╔═╡ 0b0c4650-2ce1-4879-9acd-81c16d06700e
md"""
The left plot shows the negative spike at the very beginning in more detail. In *FMIFlux.jl*, there are ready to use layers for scaling and shifting, that can automatically select appropraite parameters. These parameters are trained together with the ANN parameters by default.
"""

# ╔═╡ 0fb90681-5d04-471a-a7a8-4d0f3ded7bcf
md"""
## Introducing Gates
to control how physical and machine learning model interact

[Todo text]
[todo allow sliding between different gate openings]
"""

# ╔═╡ 95e14ea5-d82d-4044-8c68-090d74d95a61
md"""
There are basically two ways of connecting two blocks (the ANN and the FMU):
- In **series**, so one block is getting signals from the other block and is able to *manipulate* or *correct* this signals. This way, modeling errors can be corrected.
- In **parallel**, so both are getting the same signals and calculate own outputs, these outputs must be merged afterwards. This way, additional system parts, like e.g. forces or momentum, can be learned and added.

The good news is, you don't have to decide this beforehand. This is something that the optimizer can dacide, if we interoduce parameters, that allow for both topologies. This structure is named *gates*.
"""

# ╔═╡ cbae6aa4-1338-428c-86aa-61d3304e33ed
@bind GATE_INIT_FMU Slider(0.0:0.1:1.0, default=1.0)

# ╔═╡ 2fa1821b-aaec-4de4-bfb4-89560790dc39
md"""
Change the opening of the **FMU gate** $(GATE_INIT_FMU) for dα1:
"""

# ╔═╡ 8c56acd6-94d3-4cbc-bc29-d249740268a0
@bind GATE_INIT_ANN Slider(0.0:0.1:1.0, default=0.0)

# ╔═╡ 9b52a65a-f20c-4387-aaca-5292a92fb639
md"""
Change the opening of the **ANN gate** $(GATE_INIT_ANN) for dα1:
"""

# ╔═╡ 845a95c4-9a35-44ae-854c-57432200da1a
md"""
The FMU gate value for dα1 is $(GATE_INIT_FMU) and the ANN gate value is $(GATE_INIT_ANN). This means the hybrid model dα1 is composed of $(GATE_INIT_FMU*100)% of dα1 from the FMU and of $(GATE_INIT_ANN*100)% of dα1 from the ANN.
"""

# ╔═╡ fd1cebf1-5ccc-4bc5-99d4-1eaa30e9762e
md"""
This equals the serial topology: $((GATE_INIT_FMU==0 && GATE_INIT_ANN==1)) $br
This equals the parallel topology: $((GATE_INIT_FMU==1 && GATE_INIT_ANN==1))
"""

# ╔═╡ 5a399a9b-32d9-4f93-a41f-8f16a4b102dc
begin 
	function build_model_gates()
		Random.seed!(123)
		
		cache = CacheLayer()                        # allocate a cache layer
		cacheRetrieve = CacheRetrieveLayer(cache)   # allocate a cache retrieve layer, link it to the cache layer
	
	    # we have two signals (acceleration, consumption) and two sources (ANN, FMU), so four gates:
	    # (1) acceleration from FMU (gate=1.0 | open)
	    # (2) consumption  from FMU (gate=1.0 | open)
	    # (3) acceleration from ANN (gate=0.0 | closed)
	    # (4) consumption  from ANN (gate=0.0 | closed)
	    # the acelerations [1,3] and consumptions [2,4] are paired
	    gates = ScaleSum([GATE_INIT_FMU, GATE_INIT_ANN], [[1,2]]) # gates with sum
	
	    # setup the NeuralFMU topology
	    model_gates = Flux.f64(Chain(dx -> cache(dx),                    # cache `dx`
	                  Dense(1, 16, tanh),  
						Dense(16, 1, tanh),  # pre-process `dx`
	                  dx -> cacheRetrieve(1, dx),       # dynamics FMU | dynamics ANN
	                  gates))       # stack toget

		model_input = collect([v] for v in data_train.da1)
		model_output = collect(model_gates(inp) for inp in model_input)
		ANN_output = collect(model_gates[2:3](inp) for inp in model_input)
		
		fig = plot(; ylims=(-3,1), legend=:bottomright) 
		plot!(fig, data_train.t, collect(v[1] for v in model_input), label="dα1 of FMU")
		plot!(fig, data_train.t, collect(v[1] for v in ANN_output), label="dα1 of ANN")
		plot!(fig, data_train.t, collect(v[1] for v in model_output), label="dα1 of NeuralFMU")
		
		return fig
	end
	build_model_gates()
end

# ╔═╡ 2a5157c5-f5a2-4330-b2a3-0c1ec0b7adff
md"""
# Building the NeuralFMU
**... putting everything together**


"""

# ╔═╡ 4454c8d2-68ed-44b4-adfa-432297cdc957
md"""
## FMU inputs
In general, you can use arbitrary values as input for the FMU layer, like system inputs, states or parameters. In this example, we want to use only system states as inputs for the FMU layer.

**FMU layer inputs**
[Todo state names]

To preserve the ODE topology (a mapping from state to state derivative), we use all system state derivatives as layer outputs. However, you can choose further outputs if you want to... and you definitely should.

## ANN inputs
[Todo derivative 3, 5 names]

Pick additional ANN layer inputs:
"""

# ╔═╡ d240c95c-5aba-4b47-ab8d-2f9c0eb854cd
@bind y_refs MultiCheckBox([STATE_A1 => "Angle Joint 1", STATE_A2 => "Angle Joint 2", VAR_TCP_PX => "TCP position x", VAR_TCP_PY => "TCP position y", VAR_TCP_VX => "TCP velocity x", VAR_TCP_VY => "TCP velocity y", VAR_TCP_F => "TCP (normal) force z"])

# ╔═╡ 06937575-9ab1-41cd-960c-7eef3e8cae7f
md"""
It might be clever to pick additional inputs, because the effect being learned (slip-stick of the pen) might depend on this additional input. However, every additional signal has a little negative impact on the computational performance.
"""

# ╔═╡ 356b6029-de66-418f-8273-6db6464f9fbf
md"""
## ANN size
"""

# ╔═╡ 53e971d8-bf43-41cc-ac2b-20dceaa78667
@bind GATES_INIT Slider(0.0:0.1:1.0, default=0.0)

# ╔═╡ 5805a216-2536-44ac-a702-d92e86d435a4
md"""
The ANN shall have $(@bind NUM_LAYERS Select([2, 3, 4])) layers with a width of $(@bind LAYERS_WIDTH Select([8, 16, 32])) each.

The gates shall be initialized with $(GATES_INIT), slide to change:
"""

# ╔═╡ e8b8c63b-2ca4-4e6a-a801-852d6149283e
md"""
All gates shall be initialized with $(GATES_INIT), meaning the ANN contributes $(GATES_INIT*100)% to the hybrid model derivatives, while the FMU contributes $(100-GATES_INIT*100)%. These parameters are adapted during training, these are only start values.
"""

# ╔═╡ c0ac7902-0716-4f18-9447-d18ce9081ba5
md"""
## Resulting NeuralFMU
Your final NeuralFMU topology:
"""

# ╔═╡ 84215a73-1ab0-416d-a9db-6b29cd4f5d2a
begin 

function build_topology()

	ANN_input_Vars = [recordValues[1:2]..., y_refs...]
	ANN_input_Vals = fmiGetSolutionValue(sol_fmu_train, ANN_input_Vars)
	ANN_input_Idcs = [4, 6]
	for i in 1:length(y_refs)
		push!(ANN_input_Idcs, i+6)
	end

	GATE_INIT = 1.0

	# pre- and post-processing
    preProcess = ShiftScale(ANN_input_Vals)         # we put in the derivatives recorded above, FMIFlux shift and scales so we have a data mean of 0 and a standard deivation of 1
    #preProcess.scale[:] *= 0.1                         # add some additional "buffer"
    postProcess = ScaleShift(preProcess; indices=[1,2])   # initialize the postPrcess as inverse of the preProcess, but only take indices 2 and 3 (we don't need 1, the vehcile velocity)

    # cache
    cache = CacheLayer()                        # allocate a cache layer
    cacheRetrieve = CacheRetrieveLayer(cache)   # allocate a cache retrieve layer, link it to the cache layer

	gates = ScaleSum([1.0-GATE_INIT, 1.0-GATE_INIT, GATE_INIT, GATE_INIT], [[1,3], [2,4]]) # gates with sum
	
	ANN_layers = []
	push!(ANN_layers, Dense(2+length(y_refs), LAYERS_WIDTH, tanh)) # first layer 
	for i in 3:NUM_LAYERS
		push!(ANN_layers, Dense(LAYERS_WIDTH, LAYERS_WIDTH, tanh))
	end
	push!(ANN_layers, Dense(LAYERS_WIDTH, 2, tanh)) # last layer 

	model = Flux.f64(Chain(x -> fmu(; x=x, dx_refs=:all, y_refs=y_refs), 
		dxy -> cache(dxy),                    # cache `dx`
        dxy -> dxy[ANN_input_Idcs], 
		preProcess,
		ANN_layers...,
	postProcess,
	dx -> cacheRetrieve(4, 6, dx),       # dynamics FMU | dynamics ANN
    gates,                              # compute resulting dx from ANN + FMU
      dx -> cacheRetrieve(1:3, dx[1], 5, dx[2])))

	return model
	
end

final_model = build_topology()

end

# ╔═╡ f741b213-a20d-423a-a382-75cae1123f2c
final_model(x0)

# ╔═╡ f02b9118-3fb5-4846-8c08-7e9bbca9d208
md"""
On basis of this `Chain`, we can build a NeuralFMU very easy:
"""

# ╔═╡ 91473bef-bc23-43ed-9989-34e62166d455
neuralFMU = ME_NeuralFMU(fmu,                 # the FMU used in the NeuralFMU 
                    	 final_model,         # the model we specified above 
                         (tStart, tStop), # a default start and stop time for solving the NeuralFMU
                    	 solver; # the solver
                         saveat=tSave)  # time points to save the solution at

# ╔═╡ d347d51b-743f-4fec-bed7-6cca2b17bacb
md"""
# Training

After setting everything up, we can give it a try and train our created NeuralFMU. Deepending on the chosen optimization hyper parameters, this will be more or les successful. Feel free to play around a bit, but keep in mind that for real application design, you should do hyper parameter optimization instead of playing around by yourself - I did this time-wasting mistake for too long.
"""

# ╔═╡ d60d2561-51a4-4f8a-9819-898d70596e0c
md"""
## Hyperparameters
Besides the already introduces hyperparameters - the depth, width and initial gate opening off the hybrid model - further parameters might have significant impact on the training success.

### Optimizer
For this example, we use the well-known `Adam`-Optimizer with a step size `eta` of $(@bind ETA Select([1e-4 => "1e-4", 1e-3 => "1e-3", 1e-2 => "1e-2"])). 

### Batching 
Because data has a significant length, gradient computation over the entire simulation trajectory might not be effective. The most common approach is to *cut* data into slices and train on these instead of the entire trajctory at once. In this example, data is cut in pieces with length of $(@bind BATCHDUR Select([0.05, 0.1, 0.15, 0.2])) seconds.
"""

# ╔═╡ c97f2dea-cb18-409d-9ae8-1d03647a6bb3
md"""
This results in a batch with $(round(Integer, data_train.t[end] / BATCHDUR)) elements.
"""

# ╔═╡ 366abd1a-bcb5-480d-b1fb-7c76930dc8fc
md"""
We use a simple `Random` scheduler here, that picks a random batch element for the next training step. Other schedulers are pre-implemented in FMIFlux.jl.
"""

# ╔═╡ 7e2ffd6f-19b0-435d-8e3c-df24a591bc55
md"""
### Loss Function
Different loss functions are thinkable here. Two quantities that should be considered are the motor currents and the motor revolution speeds. For this workshop we use the *Mean Average Error* (MAE) over the motor currents. Other loss functions can easily be deployed.
"""

# ╔═╡ caa5e04a-2375-4c56-8072-52c140adcbbb
function _loss(solution::FMU2Solution, data::FMIZoo.RobotRR_Data)

    # determine the start/end indices `ts` and `te` in the data array (sampled with 100Hz)
	dt = 0.01
    ts = 1+round(Integer, solution.states.t[1] / dt)
    te = 1+round(Integer, solution.states.t[end] / dt)
    
    # retrieve the data from NeuralFMU ("where we are") and data from measurements ("where we want to be")
    i1_value = fmiGetSolutionState(solution, STATE_I1)
    i2_value = fmiGetSolutionState(solution, STATE_I2)
    i1_data = data.i1[ts:te]
    i2_data = data.i2[ts:te]
    
    Δvalue = 0.0
    Δvalue += FMIFlux.Losses.mae(i1_value, i1_data)
    Δvalue += FMIFlux.Losses.mae(i2_value, i2_data)
    
    return Δvalue
end

# ╔═╡ 69657be6-6315-4655-81e2-8edef7f21e49
md"""
The loss function value of the plain FMU is $(round(_loss(sol_fmu_train, data_train); digits=6)).
"""

# ╔═╡ 23ad65c8-5723-4858-9abe-750c3b65c28a
md"""
## Start
[todo]

Batching takes a few seconds and training a few minutes (depending on the number of training steps), this is not triggered automatically. If you are ready to go, choose a number of training steps and clikc the button. This will start a training of $(@bind STEPS Select([0, 10, 100, 1000])) training steps.
"""

# ╔═╡ e8bae97d-9f90-47d2-9263-dc8fc065c3d0
md"""
The roughly estimated training time is $(round(Integer, STEPS*2.5*BATCHDUR + 0.6/BATCHDUR)) seconds (Windows, i7 @ 3.6GHz).

**Enable Live Training** $(@bind LIVE_TRAIN CheckBox()) 
"""

# ╔═╡ 2dce68a7-27ec-4ffc-afba-87af4f1cb630
begin

function train(eta, batchdur, steps)

	if steps == 0
		return "Number of training steps is `0`, no training."
	end
	
	train_t = data_train.t
	train_data = collect([data_train.i2[i], data_train.i1[i]] for i in 1:length(train_t))

	@info "Started batching ..."
	batch = batchDataSolution(neuralFMU, # our NeuralFMU model
                              t -> FMIZoo.getState(data_train, t), # a function returning a start state for a given time point `t`, to determine start states for batch elements
                              train_t, # data time points
                              train_data; # data cumulative consumption 
                              batchDuration=batchdur, # duration of one batch element
                              indicesModel=[1,2], # model indices to train on (1 and 2 equal the `electrical current` states)
                              plot=false, # don't show intermediate plots (try this outside of Pluto)
                              showProgress=false,
                              parameters=parameters)   
	
	@info "... batching finished!"

	# a random element scheduler
	scheduler = RandomScheduler(neuralFMU, batch; applyStep=1, plotStep=0)

	lossFct = (solution::FMU2Solution) -> _loss(solution, data_train)

	loss = p -> FMIFlux.Losses.loss(neuralFMU, # the NeuralFMU to simulate
                                    batch; # the batch to take an element from
                                    p=p, # the NeuralFMU training parameters (given as input)
                                    lossFct=lossFct, # our custom loss function
                                    batchIndex=scheduler.elementIndex, # the index of the batch element to take, determined by the choosen scheduler
                                    logLoss=true, # log losses after every evaluation
                                    showProgress=false,
									parameters=parameters) 

	params = FMIFlux.params(neuralFMU)

	FMIFlux.initialize!(scheduler; p=params[1], showProgress=false, parameters=parameters)

	BETA1 = 0.9
	BETA2 = 0.999
	optim = Adam(eta, (BETA1, BETA2))

	@info "Started training ..."

	 FMIFlux.train!(loss, # the loss function for training
                   neuralFMU, # the parameters to train
                   Iterators.repeated((), steps), # an iterator repeating `steps` times
                   optim; # the optimizer to train
                   gradient=:ReverseDiff, # use ReverseDiff, because it's much faster!
                   cb=() -> FMIFlux.update!(scheduler), # update the scheduler after every step 
                   proceed_on_assert=false) # go on if a training steps fails (e.g. because of instability)  

	@info "... training finished!"
end

"👻 Hidden Code | You probably want to skip this code section on the first run."

end

# ╔═╡ ac7f8003-1d8d-428f-a04d-3dcdf96099ca
md""" 
Training results:
"""

# ╔═╡ c3f5704b-8e98-4c46-be7a-18ab4f139458
let
	if LIVE_TRAIN
		train(ETA, BATCHDUR, STEPS)
	else
		LIVE_TRAIN_MESSAGE
	end
end

# ╔═╡ 27458e32-5891-4afc-af8e-7afdf7e81cc6
begin

function plotPaths!(fig, t, x, N; color=:black, label=:none, kwargs...)
    paths = []
    path = nothing
    lastN = N[1]
    for i in 1:length(N)
        if N[i] == 0.0
            if lastN == 1.0
                push!(path, (t[i], x[i]) )
                push!(paths, path)
            end
        end

        if N[i] == 1.0
            if lastN == 0.0
                path = []
            end
            push!(path, (t[i], x[i]) )
        end

        lastN = N[i]
    end
    if length(path) > 0
        push!(paths, path)
    end

    isfirst = true
    for path in paths
        plot!(fig, collect(v[1] for v in path), collect(v[2] for v in path); 
            label=isfirst ? label : :none, 
            color=color, 
            kwargs...)
        isfirst = false
    end

    return fig
end

HIDDEN_CODE_MESSAGE

end

# ╔═╡ ff106912-d18c-487f-bcdd-7b7af2112cab
md"""
# Results 

**Enable Live Results** $(@bind LIVE_RESULTS CheckBox()) 

## Training results
Let's check out the *training* results of the freshly trained NeuralFMU.
"""

# ╔═╡ 5dd491a4-a8cd-4baf-96f7-7a0b850bb26c
begin
	if LIVE_RESULTS
		fmu_train = fmiSimulate(fmu, (data_train.t[1], data_train.t[end]); x0=x0,
        parameters=Dict{String, Any}("fileName" => data_train.params["fileName"]), 
        recordValues=["rRPositionControl_Elasticity.tCP.p_x", 
                      "rRPositionControl_Elasticity.tCP.p_y",
                      "rRPositionControl_Elasticity.tCP.N"],
        showProgress=true, maxiters=1e7, saveat=data_train.t, solver=Tsit5());
	else
		LIVE_RESULTS_MESSAGE
	end
end

# ╔═╡ 1195a30c-3b48-4bd2-8a3a-f4f74f3cd864
begin
	if LIVE_RESULTS
		result_train = neuralFMU(x0, (data_train.t[1], data_train.t[end]); 
        parameters=Dict{String, Any}("fileName" => data_train.params["fileName"]), 
        recordValues=["rRPositionControl_Elasticity.tCP.p_x", 
                      "rRPositionControl_Elasticity.tCP.p_y", 
                      "rRPositionControl_Elasticity.tCP.N"],
        showProgress=true, maxiters=1e7, saveat=data_train.t);
	else
		LIVE_RESULTS_MESSAGE
	end
end

# ╔═╡ 919419fe-35de-44bb-89e4-8f8688bee962
let
	if LIVE_RESULTS
		fig = plot(; dpi=300, size=(200*3,60*3))
		plotPaths!(fig, data_train.tcp_px, data_train.tcp_py, data_train.tcp_norm_f, label="Data", color=:black, style=:dash)
		plotPaths!(fig, collect(v[1] for v in fmu_train.values.saveval), collect(v[2] for v in fmu_train.values.saveval), collect(v[3] for v in fmu_train.values.saveval), label="FMU", color=:orange)
		plotPaths!(fig, collect(v[1] for v in result_train.values.saveval), collect(v[2] for v in result_train.values.saveval), collect(v[3] for v in result_train.values.saveval), label="NeuralFMU", color=:blue)
	else
		LIVE_RESULTS_MESSAGE
	end
end

# ╔═╡ 2918daf2-6499-4019-a04b-8c3419ee1ab7
let
	if LIVE_RESULTS
		fig = plot(; dpi=300, size=(40*10,40*10), xlims=(0.165, 0.205), ylims=(-0.035, 0.005))
		plotPaths!(fig, data_train.tcp_px, data_train.tcp_py, data_train.tcp_norm_f, label="Data", color=:black, style=:dash)
		plotPaths!(fig, collect(v[1] for v in fmu_train.values.saveval), collect(v[2] for v in fmu_train.values.saveval), collect(v[3] for v in fmu_train.values.saveval), label="FMU", color=:orange)
		plotPaths!(fig, collect(v[1] for v in result_train.values.saveval), collect(v[2] for v in result_train.values.saveval), collect(v[3] for v in result_train.values.saveval), label="NeuralFMU", color=:blue)
	else 
		LIVE_RESULTS_MESSAGE
	end
end

# ╔═╡ 048e39c3-a3d9-4e6b-b050-1fd5a919e4ae
let
	if LIVE_RESULTS
		fig = plot(; dpi=300, size=(50*10,40*10), xlims=(0.245, 0.295), ylims=(-0.04, 0.0))
		plotPaths!(fig, data_train.tcp_px, data_train.tcp_py, data_train.tcp_norm_f, label="Data", color=:black, style=:dash)
		plotPaths!(fig, collect(v[1] for v in fmu_train.values.saveval), collect(v[2] for v in fmu_train.values.saveval), collect(v[3] for v in fmu_train.values.saveval), label="FMU", color=:orange)
		plotPaths!(fig, collect(v[1] for v in result_train.values.saveval), collect(v[2] for v in result_train.values.saveval), collect(v[3] for v in result_train.values.saveval), label="NeuralFMU", color=:blue)
	else
		LIVE_RESULTS_MESSAGE 
	end
end

# ╔═╡ b489f97d-ee90-48c0-af06-93b66a1f6d2e
md"""
## Validation results
Let's check out the *validation* results of the freshly trained NeuralFMU.
"""

# ╔═╡ ea0ede8d-7c2c-4e72-9c96-3260dc8d817d
begin
	if LIVE_RESULTS
		fmu_validation = fmiSimulate(fmu, (data_validation.t[1], data_validation.t[end]); x0=x0,
	        parameters=Dict{String, Any}("fileName" => data_validation.params["fileName"]), 
	        recordValues=["rRPositionControl_Elasticity.tCP.p_x", 
	                      "rRPositionControl_Elasticity.tCP.p_y",
	                      "rRPositionControl_Elasticity.tCP.N"],
	        showProgress=true, maxiters=1e7, saveat=data_validation.t, solver=Tsit5());
	else 
		LIVE_RESULTS_MESSAGE 
	end
end

# ╔═╡ 51aed933-2067-4ea8-9c2f-9d070692ecfc
begin
	if LIVE_RESULTS
		result_validation = neuralFMU(x0, (data_validation.t[1], data_validation.t[end]); 
        parameters=Dict{String, Any}("fileName" => data_validation.params["fileName"]), 
        recordValues=["rRPositionControl_Elasticity.tCP.p_x", 
            "rRPositionControl_Elasticity.tCP.p_y",
            "rRPositionControl_Elasticity.tCP.N"],
        showProgress=true, maxiters=1e7, saveat=data_validation.t);
	else
		LIVE_RESULTS_MESSAGE
	end
end

# ╔═╡ 74ef5a39-1dd7-404a-8baf-caa1021d3054
let
	if LIVE_RESULTS
		fig = plot(; dpi=300, size=(200*3,40*3))
		plotPaths!(fig, data_validation.tcp_px, data_validation.tcp_py, data_validation.tcp_norm_f, label="Data", color=:black, style=:dash)
		plotPaths!(fig, collect(v[1] for v in fmu_validation.values.saveval), collect(v[2] for v in fmu_validation.values.saveval), collect(v[3] for v in fmu_validation.values.saveval), label="FMU", color=:orange)
		plotPaths!(fig, collect(v[1] for v in result_validation.values.saveval), collect(v[2] for v in result_validation.values.saveval), collect(v[3] for v in result_validation.values.saveval), label="NeuralFMU", color=:blue)
	else
		LIVE_RESULTS_MESSAGE
	end
end

# ╔═╡ 05281c4f-dba8-4070-bce3-dc2f1319902e
let
	if LIVE_RESULTS
		fig = plot(; dpi=300, size=(35*10,50*10), xlims=(0.188, 0.223), ylims=(-0.025, 0.025))
		plotPaths!(fig, data_validation.tcp_px, data_validation.tcp_py, data_validation.tcp_norm_f, label="Data", color=:black, style=:dash)
		plotPaths!(fig, collect(v[1] for v in fmu_validation.values.saveval), collect(v[2] for v in fmu_validation.values.saveval), collect(v[3] for v in fmu_validation.values.saveval), label="FMU", color=:orange)
		plotPaths!(fig, collect(v[1] for v in result_validation.values.saveval), collect(v[2] for v in result_validation.values.saveval), collect(v[3] for v in result_validation.values.saveval), label="NeuralFMU", color=:blue)
	else
		LIVE_RESULTS_MESSAGE
	end
end

# ╔═╡ 67cfe7c5-8e62-4bf0-996b-19597d5ad5ef
let
	if LIVE_RESULTS
		fig = plot(; dpi=300, size=(25*10,50*10), xlims=(0.245, 0.27), ylims=(-0.025, 0.025), legend=:topleft)
		plotPaths!(fig, data_validation.tcp_px, data_validation.tcp_py, data_validation.tcp_norm_f, label="Data", color=:black, style=:dash)
		plotPaths!(fig, collect(v[1] for v in fmu_validation.values.saveval), collect(v[2] for v in fmu_validation.values.saveval), collect(v[3] for v in fmu_validation.values.saveval), label="FMU", color=:orange)
		plotPaths!(fig, collect(v[1] for v in result_validation.values.saveval), collect(v[2] for v in result_validation.values.saveval), collect(v[3] for v in result_validation.values.saveval), label="NeuralFMU", color=:blue)
	else
		LIVE_RESULTS_MESSAGE
	end
end

# ╔═╡ 88884204-79e4-4412-b861-ebeb5f6f7396
md""" 
# Conclusion
[TODO]

## Citation
If you find this workshop useful for your own work and/or research, please cite our related publication:

Tobias Thummerer, Johannes Stoljar and Lars Mikelsons. 2022. **NeuralFMU: presenting a workflow for integrating hybrid neuralODEs into real-world applications.** Electronics 11, 19, 3202. DOI: 10.3390/electronics11193202

## Acknowlegments
- the FMU was created using the excellent Modelica library *Servomechanisms* $br (https://github.com/afrhu/Servomechanisms)
- the linked YouTube video in the introduction is by *Alexandru Babaian* $br (https://www.youtube.com/watch?v=ryIwLLr6yRA)
"""

# ╔═╡ Cell order:
# ╟─1470df0f-40e1-45d5-a4cc-519cc3b28fb8
# ╟─7d694be0-cd3f-46ae-96a3-49d07d7cf65a
# ╟─6fc16c34-c0c8-48ce-87b3-011a9a0f4e7c
# ╟─89a683f9-4e69-4daf-80bd-486f3a135797
# ╟─8a82d8c7-b781-4600-8780-0a0a003b676c
# ╠═a1ee798d-c57b-4cc3-9e19-fb607f3e1e43
# ╠═72604eef-5951-4934-844d-d2eb7eb0292c
# ╠═21104cd1-9fe8-45db-9c21-b733258ff155
# ╠═eaae989a-c9d2-48ca-9ef8-fd0dbff7bcca
# ╠═98c608d9-c60e-4eb6-b611-69d2ae7054c9
# ╠═9d9e5139-d27e-48c8-a62e-33b2ae5b0086
# ╟─5cb505f7-01bd-4824-8876-3e0f5a922fb7
# ╠═45c4b9dd-0b04-43ae-a715-cd120c571424
# ╟─1e9541b8-5394-418d-8c27-2831951c538d
# ╠═bc077fdd-069b-41b0-b685-f9513d283346
# ╟─7b82e333-e133-4b7b-8951-5a8b6a0155db
# ╠═1bb3d6ae-af9c-4b5d-b2e7-19cee7750347
# ╟─44500f0a-1b89-44af-b135-39ce0fec5810
# ╟─33223393-bfb9-4e9a-8ea6-a3ab6e2f22aa
# ╟─915e4601-12cc-4b7e-b2fe-574e116f3a92
# ╟─f8e40baa-c1c5-424a-9780-718a42fd2b67
# ╠═74289e0b-1292-41eb-b13b-a4a5763c72b0
# ╟─f111e772-a340-4217-9b63-e7715f773b2c
# ╟─92ad1a99-4ad9-4b69-b6f3-84aab49db54f
# ╠═909de9f1-2aca-4bf0-ba60-d3418964ba4a
# ╟─d8ca5f66-4f55-48ab-a6c9-a0be662811d9
# ╠═41b1c7cb-5e3f-4074-a681-36dd2ef94454
# ╠═8f45871f-f72a-423f-8101-9ce93e5a885b
# ╠═57c039f7-5b24-4d63-b864-d5f808110b91
# ╟─8d93a1ed-28a9-4a77-9ac2-5564be3729a5
# ╠═4a8de267-1bf4-42c2-8dfe-5bfa21d74b7e
# ╟─6a8b98c9-e51a-4f1c-a3ea-cc452b9616b7
# ╟─dbde2da3-e3dc-4b78-8f69-554018533d35
# ╠═d42d0beb-802b-4d30-b5b8-683d76af7c10
# ╟─3756dd37-03e0-41e9-913e-4b4f183d8b81
# ╠═2f83bc62-5a54-472a-87a2-4ddcefd902b6
# ╠═f168b997-355d-4d01-9f18-caf3d77194a5
# ╟─dd0bdcf5-37a4-4004-934a-d535dd86c27e
# ╠═276ef0b2-568b-4e85-9b56-88170336d30c
# ╠═634f923a-5e09-42c8-bac0-bf165ab3d12a
# ╠═8aed89cf-787d-4f07-a67a-85fe384caa5e
# ╠═0c9493c4-322e-41a0-9ec7-2e2c54ae1373
# ╠═25e55d1c-388f-469d-99e6-2683c0508693
# ╟─c228eb10-d694-46aa-b952-01d824879287
# ╟─74c519c9-0eef-4798-acff-b11044bb4bf1
# ╟─786c4652-583d-43e9-a101-e28c0b6f64e4
# ╟─b42bf3d8-e70c-485c-89b3-158eb25d8b25
# ╟─5d688c3d-b5e3-4a3a-9d91-0896cc001000
# ╠═2e08df84-a468-4e99-a277-e2813dfeae5c
# ╟─0a7955e7-7c1a-4396-9613-f8583195c0a8
# ╠═4912d9c9-d68d-4afd-9961-5d8315884f75
# ╟─6a1a5092-6832-4ac7-9850-346e3fa28252
# ╟─a1aca180-d561-42a3-8d12-88f5a3721aae
# ╟─13ede3cd-99b1-4e65-8a18-9043db544728
# ╟─33791947-342b-4bf4-9d0a-c3979c0ad88a
# ╟─7622ed93-9c66-493b-9fba-c0d3755758a8
# ╟─432ab3ff-5336-4413-9807-45d4bf756e1f
# ╟─eaf37128-0377-42b6-aa81-58f0a815276b
# ╟─c030d85e-af69-49c9-a7c8-e490d4831324
# ╟─51c200c9-0de3-4e50-8884-49fe06158560
# ╟─0dadd112-3132-4491-9f02-f43cf00aa1f9
# ╟─5c2308d9-6d04-4b38-af3b-6241da3b6871
# ╟─bf6bf640-54bc-44ef-bd4d-b98e934d416e
# ╟─639889b3-b9f2-4a3c-999d-332851768fd7
# ╟─007d6d95-ad85-4804-9651-9ac3703d3b40
# ╟─ed1887df-5079-4367-ab04-9d02a1d6f366
# ╟─0b0c4650-2ce1-4879-9acd-81c16d06700e
# ╟─0fb90681-5d04-471a-a7a8-4d0f3ded7bcf
# ╟─95e14ea5-d82d-4044-8c68-090d74d95a61
# ╟─2fa1821b-aaec-4de4-bfb4-89560790dc39
# ╟─cbae6aa4-1338-428c-86aa-61d3304e33ed
# ╟─9b52a65a-f20c-4387-aaca-5292a92fb639
# ╟─8c56acd6-94d3-4cbc-bc29-d249740268a0
# ╟─845a95c4-9a35-44ae-854c-57432200da1a
# ╟─fd1cebf1-5ccc-4bc5-99d4-1eaa30e9762e
# ╟─5a399a9b-32d9-4f93-a41f-8f16a4b102dc
# ╟─2a5157c5-f5a2-4330-b2a3-0c1ec0b7adff
# ╟─4454c8d2-68ed-44b4-adfa-432297cdc957
# ╟─d240c95c-5aba-4b47-ab8d-2f9c0eb854cd
# ╟─06937575-9ab1-41cd-960c-7eef3e8cae7f
# ╟─356b6029-de66-418f-8273-6db6464f9fbf
# ╟─5805a216-2536-44ac-a702-d92e86d435a4
# ╟─53e971d8-bf43-41cc-ac2b-20dceaa78667
# ╟─e8b8c63b-2ca4-4e6a-a801-852d6149283e
# ╟─c0ac7902-0716-4f18-9447-d18ce9081ba5
# ╟─84215a73-1ab0-416d-a9db-6b29cd4f5d2a
# ╠═f741b213-a20d-423a-a382-75cae1123f2c
# ╟─f02b9118-3fb5-4846-8c08-7e9bbca9d208
# ╠═91473bef-bc23-43ed-9989-34e62166d455
# ╟─d347d51b-743f-4fec-bed7-6cca2b17bacb
# ╟─d60d2561-51a4-4f8a-9819-898d70596e0c
# ╟─c97f2dea-cb18-409d-9ae8-1d03647a6bb3
# ╟─366abd1a-bcb5-480d-b1fb-7c76930dc8fc
# ╟─7e2ffd6f-19b0-435d-8e3c-df24a591bc55
# ╠═caa5e04a-2375-4c56-8072-52c140adcbbb
# ╟─69657be6-6315-4655-81e2-8edef7f21e49
# ╟─23ad65c8-5723-4858-9abe-750c3b65c28a
# ╟─e8bae97d-9f90-47d2-9263-dc8fc065c3d0
# ╟─2dce68a7-27ec-4ffc-afba-87af4f1cb630
# ╟─ac7f8003-1d8d-428f-a04d-3dcdf96099ca
# ╟─c3f5704b-8e98-4c46-be7a-18ab4f139458
# ╟─27458e32-5891-4afc-af8e-7afdf7e81cc6
# ╟─ff106912-d18c-487f-bcdd-7b7af2112cab
# ╟─5dd491a4-a8cd-4baf-96f7-7a0b850bb26c
# ╟─1195a30c-3b48-4bd2-8a3a-f4f74f3cd864
# ╟─919419fe-35de-44bb-89e4-8f8688bee962
# ╟─2918daf2-6499-4019-a04b-8c3419ee1ab7
# ╟─048e39c3-a3d9-4e6b-b050-1fd5a919e4ae
# ╟─b489f97d-ee90-48c0-af06-93b66a1f6d2e
# ╟─ea0ede8d-7c2c-4e72-9c96-3260dc8d817d
# ╟─51aed933-2067-4ea8-9c2f-9d070692ecfc
# ╟─74ef5a39-1dd7-404a-8baf-caa1021d3054
# ╟─05281c4f-dba8-4070-bce3-dc2f1319902e
# ╟─67cfe7c5-8e62-4bf0-996b-19597d5ad5ef
# ╟─88884204-79e4-4412-b861-ebeb5f6f7396
