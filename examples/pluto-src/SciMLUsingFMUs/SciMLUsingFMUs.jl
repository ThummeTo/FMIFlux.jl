### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ a1ee798d-c57b-4cc3-9e19-fb607f3e1e43
using PlutoUI # Notebook UI

# ╔═╡ 72604eef-5951-4934-844d-d2eb7eb0292c
using FMI # load and simulate FMUs

# ╔═╡ 21104cd1-9fe8-45db-9c21-b733258ff155
using FMIFlux # machine learning with FMUs

# ╔═╡ 9d9e5139-d27e-48c8-a62e-33b2ae5b0086
using FMIZoo # a collection of demo FMUs

# ╔═╡ eaae989a-c9d2-48ca-9ef8-fd0dbff7bcca
using FMIFlux.Flux # default Julia Machine Learning library

# ╔═╡ 98c608d9-c60e-4eb6-b611-69d2ae7054c9
using FMIFlux.DifferentialEquations # the mighty (O)DE solver suite

# ╔═╡ ddc9ce37-5f93-4851-a74f-8739b38ab092
using ProgressLogging: @withprogress, @logprogress, @progressid, uuid4

# ╔═╡ de7a4639-e3b8-4439-924d-7d801b4b3eeb
using BenchmarkTools # default benchmarking library

# ╔═╡ 45c4b9dd-0b04-43ae-a715-cd120c571424
using Plots

# ╔═╡ 1470df0f-40e1-45d5-a4cc-519cc3b28fb8
md"""
# Scientific Machine Learning $br using Functional Mock-Up Units
(former *Hybrid Modeling using FMI*)

Workshop $br
@ JuliaCon 2024 (Eindhoven, Netherlands) $br
@ MODPROD 2024 (Linköping University, Sweden)

by Tobias Thummerer (University of Augsburg)

*#hybridmodeling, #sciml, #neuralode, #neuralfmu, #penode*

# Abstract
If there is something YOU know about a physical system, AI shouldn’t need to learn it. How to integrate YOUR system knowledge into a ML development process is the core topic of this hands-on workshop. The entire workshop evolves around a challenging use case from robotics: Modeling a robot that is able to write arbitrary messages with a pen. After introducing the topic and the considered use case, participants can experiment with their very own hybrid model topology. 

# Introduction
This workshop focuses on the integration of Functional Mock-Up Units (FMUs) into a machine learning topology. FMUs are simulation models that can be generated within a variety of modeling tools, see the [FMI homepage](https://fmi-standard.org/). Together with deep neural networks that complement and improve the FMU prediction, so called *neural FMUs* can be created. 
The workshop itself evolves around the hybrid modeling of a *Selective Compliance Assembly Robot Arm* (SCARA), that is able to write user defined words on a sheet of paper. A ready to use physical simulation model (FMU) for the SCARA is given and shortly highlighted in this workshop. However, this model – as any simulation model – shows some deviations if compared to measurements from the real system. These deviations results from not modeled slip-stick-friction: The pen sticks to the paper until a force limit is reached, but then moves jerkily. A hard to model physical effect – but not for a neural FMU.

More advanced code snippets are hidden by default and marked with a ghost `👻`. Computations, that are disabled for performance reasons, are marked with `ℹ️`. They offer a hint how to enable the idled computation by activating the corresponding checkbox marked with `🎬`. 

## Example Video
If you haven't seen such a SCARA system yet, you can watch the following video. There are many more similar videos out there.
"""

# ╔═╡ 7d694be0-cd3f-46ae-96a3-49d07d7cf65a
html"""
<iframe width="560" height="315" src="https://www.youtube.com/embed/ryIwLLr6yRA?si=ncr1IXlnuNhWPWgl" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
"""

# ╔═╡ 10cb63ad-03d7-47e9-bc33-16c7786b9f6a
md"""
This video is by *Alexandru Babaian* on YouTube.

## Workshop Video
"""

# ╔═╡ 1e0fa041-a592-42fb-bafd-c7272e346e46
html"""
<iframe width="560" height="315" src="https://www.youtube.com/embed/sQ2MXSswrSo?si=XcEoe1Ai7U6hqnp5" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
"""

# ╔═╡ 6fc16c34-c0c8-48ce-87b3-011a9a0f4e7c
md"""
This video is from JuliaCon 2024 (Eindhoven, Netherlands).

## Requirements
To follow this workshop, you should ...
- ... have a rough idea what the *Functional Mock-Up Interface* is and how the standard-conform models - the *Functional Mock-Up Units* - work. If not, a good source is the homepage of the standard, see the [FMI Homepage](https://fmi-standard.org/).
- ... know the *Julia Programming Language* or at least have some programming skills in another high-level programming language like *Python* or *Matlab*. An introduction to Julia can be found on the [Julia Homepage](https://julialang.org/), but there are many more introductions in different formats available on the internet.
- ... have an idea of how modeling (in terms of modeling ODE and DAE systems) and simulation (solving) of such models works.

The technical requirements are:

|   | recommended | minimum | your |
| ----- | ---- | ---- | ---- |
| RAM | $\geq$ 16.0GB | 8.0GB | $(round(Sys.total_memory() / 2^30; digits=1))GB |
| OS | Windows | Windows / Linux | $(Sys.islinux() ? "Linux" : (Sys.iswindows() ? "Windows" : "unsupported"))
| Julia | 1.10 | 1.6 | $("" * string(Int(VERSION.major)) * "." * string(Int(VERSION.minor))) |

This said, we can start "programming"! The entire notebook is pre-implemented, so you can use it without writing a single line of code. Users new to Julia can use interactive UI elements to interact, while more advance users can view and manipulate corresponding code. Let's go! 
"""

# ╔═╡ 8a82d8c7-b781-4600-8780-0a0a003b676c
md"""
## Loading required Julia libraries
Before starting with the actual coding, we load in the required Julia libraries. 
This Pluto-Notebook installs all required packages automatically.
However, this will take some minutes when you start the notebook for the first time... it is recommended to not interact with the UI elements as long as the first compilation runs (orange status light in the bottom right corner).
"""

# ╔═╡ a02f77d1-00d2-46a3-91ba-8a7f5b4bbdc9
md"""
First, we load the Pluto UI elements:
"""

# ╔═╡ 02f0add7-9c4e-4358-8b5e-6863bae3ee75
md"""
Then, the three FMI-libraries we need for FMU loading, machine learning and the FMU itself:
"""

# ╔═╡ 85308992-04c4-4d20-a840-6220cab54680
md"""
Some additional libraries for machine learning and ODE solvers:
"""

# ╔═╡ 3e2579c2-39ce-4249-ad75-228f82e616da
md"""
To visualize a progress bar during training:
"""

# ╔═╡ 93fab704-a8dd-47ec-ac88-13f32be99460
md"""
And to do some benchmarking:
"""

# ╔═╡ 5cb505f7-01bd-4824-8876-3e0f5a922fb7
md""" 
Load in the plotting libraries ...
"""

# ╔═╡ 33d648d3-e66e-488f-a18d-e538ebe9c000
import PlotlyJS

# ╔═╡ 1e9541b8-5394-418d-8c27-2831951c538d
md"""
... and use the beautiful `plotly` backend for interactive plots.
"""

# ╔═╡ e6e91a22-7724-46a3-88c1-315c40660290
plotlyjs()

# ╔═╡ 44500f0a-1b89-44af-b135-39ce0fec5810
md"""
Next, we define some helper functions, that are not important to follow the workshop - they are hidden by default. However they are here, if you want to explore what it takes to write fully working code. If you do this workshop for the first time, it is recommended to skip the hidden part and directly go on.
"""

# ╔═╡ 74d23661-751b-4371-bf6b-986149124e81
md"""
Display the table of contents:
"""

# ╔═╡ c88b0627-2e04-40ab-baa2-b4c1edfda0c3
TableOfContents()

# ╔═╡ 915e4601-12cc-4b7e-b2fe-574e116f3a92
md"""
# Loading Model (FMU) and Data
We want to do hybrid modeling, so we need a simulation model and some data to work with. Fortunately, someone already prepared both for us. We start by loading some data from *FMIZoo.jl*, which is a collection of FMUs and corresponding data.
"""

# ╔═╡ f8e40baa-c1c5-424a-9780-718a42fd2b67
md"""
## Training Data
First, we need some data to train our hybrid model on. We can load data for our SCARA (here called `RobotRR`) with the following line:
"""

# ╔═╡ 74289e0b-1292-41eb-b13b-a4a5763c72b0
# load training data for the `RobotRR` from the FMIZoo
data_train = FMIZoo.RobotRR(:train)

# ╔═╡ 33223393-bfb9-4e9a-8ea6-a3ab6e2f22aa
begin

    # define the printing messages used at different places in this notebook
    LIVE_RESULTS_MESSAGE =
        md"""ℹ️ Live plotting is disabled to safe performance. Checkbox `Plot Results`."""
    LIVE_TRAIN_MESSAGE =
        md"""ℹ️ Live training is disabled to safe performance. Checkbox `Start Training`."""
    BENCHMARK_MESSAGE =
        md"""ℹ️ Live benchmarks are disabled to safe performance. Checkbox `Start Benchmark`."""
    HIDDEN_CODE_MESSAGE =
        md"""> 👻 Hidden Code | You probably want to skip this code section on the first run."""

    import FMI.FMIImport.FMICore: hasCurrentComponent, getCurrentComponent, FMU2Solution
    import Random

    function fmiSingleInstanceMode!(
        fmu::FMU2,
        mode::Bool,
        params = FMIZoo.getParameter(data_train, 0.0; friction = false),
        x0 = FMIZoo.getState(data_train, 0.0),
    )

        fmu.executionConfig = deepcopy(FMU2_EXECUTION_CONFIGURATION_NO_RESET)

        # for this model, state events are generated but don't need to be handled,
        # we can skip that to gain performance
        fmu.executionConfig.handleStateEvents = false

        fmu.executionConfig.loggingOn = false
        #fmu.executionConfig.externalCallbacks = true

        if mode
            # switch to a more efficient execution configuration, allocate only a single FMU instance, see:
            # https://thummeto.github.io/FMI.jl/dev/features/#Execution-Configuration
            fmu.executionConfig.terminate = true
            fmu.executionConfig.instantiate = false
            fmu.executionConfig.reset = true
            fmu.executionConfig.setup = true
            fmu.executionConfig.freeInstance = false
            c, _ = FMIFlux.prepareSolveFMU(
                fmu,
                nothing,
                fmu.type,
                true, # instantiate 
                false, # free 
                true, # terminate 
                true, # reset 
                true, # setup
                params;
                x0 = x0,
            )
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
            FMIFlux.finishSolveFMU(
                fmu,
                c,
                true, # free 
                true,
            ) # terminate
        end
        return nothing
    end

    function prepareSolveFMU(fmu, parameters)
        FMIFlux.prepareSolveFMU(
            fmu,
            nothing,
            fmu.type,
            fmu.executionConfig.instantiate,
            fmu.executionConfig.freeInstance,
            fmu.executionConfig.terminate,
            fmu.executionConfig.reset,
            fmu.executionConfig.setup,
            parameters,
        )
    end

    function dividePath(values)
        last_value = values[1]
        paths = []
        path = []
        for j = 1:length(values)
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
        i = 1 + round(Integer, t / dt)
        v = solution.values.saveval[i]

        l1 = 0.2
        l2 = 0.1

        margin = 0.05
        scale = 1500
        fig = plot(;
            title = "Time $(round(t; digits=1))s",
            size = (
                round(Integer, (2 * margin + l1 + l2) * scale),
                round(Integer, (l1 + l2 + 2 * margin) * scale),
            ),
            xlims = (-margin, l1 + l2 + margin),
            ylims = (-l1 - margin, l2 + margin),
            legend = :bottomleft,
        )

        p0 = [0.0, 0.0]
        p1 = p0 .+ [cos(a1) * l1, sin(a1) * l1]
        p2 = p1 .+ [cos(a1 + a2) * l2, sin(a1 + a2) * l2]

        f_norm = collect(v[3] for v in solution.values.saveval)

        paths = dividePath(f_norm)
        drawing = collect(v[1:2] for v in solution.values.saveval)
        for path in paths
            plot!(
                fig,
                collect(v[1] for v in drawing[path]),
                collect(v[2] for v in drawing[path]),
                label = :none,
                color = :black,
                style = :dot,
            )
        end

        paths = dividePath(f_norm[1:i])
        drawing_is = collect(v[4:5] for v in solution.values.saveval)[1:i]
        for path in paths
            plot!(
                fig,
                collect(v[1] for v in drawing_is[path]),
                collect(v[2] for v in drawing_is[path]),
                label = :none,
                color = :green,
                width = 2,
            )
        end

        plot!(fig, [p0[1], p1[1]], [p0[2], p1[2]], label = :none, width = 3, color = :blue)
        plot!(fig, [p1[1], p2[1]], [p1[2], p2[2]], label = :none, width = 3, color = :blue)

        scatter!(
            fig,
            [p0[1]],
            [p0[2]],
            label = "R1 | α1=$(round(a1; digits=3)) rad",
            color = :red,
        )
        scatter!(
            fig,
            [p1[1]],
            [p1[2]],
            label = "R2 | α2=$(round(a2; digits=3)) rad",
            color = :purple,
        )

        scatter!(fig, [v[1]], [v[2]], label = "TCP | F=$(v[3]) N", color = :orange)
    end

    HIDDEN_CODE_MESSAGE

end # begin

# ╔═╡ 92ad1a99-4ad9-4b69-b6f3-84aab49db54f
@bind t_train_plot Slider(0.0:0.1:data_train.t[end], default = data_train.t[1])

# ╔═╡ f111e772-a340-4217-9b63-e7715f773b2c
md"""
Let's have a look on the data! It's the written word *train*.
You can use the slider to pick a specific point in time to plot the "robot" as recorded as part of the data. 

The current picked time is $(round(t_train_plot; digits=1))s.
"""

# ╔═╡ 909de9f1-2aca-4bf0-ba60-d3418964ba4a
plotRobot(data_train.solution, t_train_plot)

# ╔═╡ d8ca5f66-4f55-48ab-a6c9-a0be662811d9
md"""
> 👁️ Interestingly, the first part of the word "trai" is not significantly affected by the slip-stick-effect, the actual TCP trajectory (green) lays quite good on the target position (black dashed). However, the "n" is very jerky. This can be explained by the increasing lever, the motor needs more torque to overcome the static friction the further away the TCP (orange) is from the robot base (red).

Let's extract a start and stop time, as well as saving points for the later solving process:
"""

# ╔═╡ 41b1c7cb-5e3f-4074-a681-36dd2ef94454
tSave = data_train.t # time points to save the solution at

# ╔═╡ 8f45871f-f72a-423f-8101-9ce93e5a885b
tStart = tSave[1]    # start time for simulation of FMU and neural FMU

# ╔═╡ 57c039f7-5b24-4d63-b864-d5f808110b91
tStop = tSave[end]   # stop time for simulation of FMU and neural FMU

# ╔═╡ 4510022b-ad28-4fc2-836b-e4baf3c14d26
md"""
Finally, also the start state can be grabbed from *FMIZoo.jl*, as well as some default parameters for the simulation model we load in the next section. How to interpret the six states is discussed in the next section where the model is loaded.
"""

# ╔═╡ 9589416a-f9b3-4b17-a381-a4f660a5ee4c
x0 = FMIZoo.getState(data_train, tStart)

# ╔═╡ 326ae469-43ab-4bd7-8dc4-64575f4a4d3e
md"""
The parameter array only contains the path to the training data file, the trajectory writing "train".
"""

# ╔═╡ 8f8f91cc-9a92-4182-8f18-098ae3e2c553
parameters = FMIZoo.getParameter(data_train, tStart; friction = false)

# ╔═╡ 8d93a1ed-28a9-4a77-9ac2-5564be3729a5
md"""
## Validation Data
To check whether the hybrid model was not only able to *imitate*, but *understands* the training data, we need some unknown data for validation. In this case, the written word "validate".
"""

# ╔═╡ 4a8de267-1bf4-42c2-8dfe-5bfa21d74b7e
# load validation data for the `RobotRR` from the FMIZoo
data_validation = FMIZoo.RobotRR(:validate)

# ╔═╡ dbde2da3-e3dc-4b78-8f69-554018533d35
@bind t_validate_plot Slider(0.0:0.1:data_validation.t[end], default = data_validation.t[1])

# ╔═╡ 6a8b98c9-e51a-4f1c-a3ea-cc452b9616b7
md"""
Let's have a look on the validation data!
Again, you can use the slider to pick a specific point in time. 

The current time is $(round(t_validate_plot; digits=1))s.
"""

# ╔═╡ d42d0beb-802b-4d30-b5b8-683d76af7c10
plotRobot(data_validation.solution, t_validate_plot)

# ╔═╡ e50d7cc2-7155-42cf-9fef-93afeee6ffa4
md"""
> 👁️ It looks similar to the effect we know from training data, the first part "valida" is not significantly affected by the slip-stick-effect, but the "te" is very jerky. Again, think of the increasing lever ...
"""

# ╔═╡ 3756dd37-03e0-41e9-913e-4b4f183d8b81
md"""
## Simulation Model (FMU)
The SCARA simulation model is called `RobotRR` for `Robot Rotational Rotational`, indicating that this robot consists of two rotational joints, connected by links. It is loaded with the following line of code:
"""

# ╔═╡ 2f83bc62-5a54-472a-87a2-4ddcefd902b6
# load the FMU named `RobotRR` from the FMIZoo
# the FMU was exported from Dymola (version 2023x)
# load the FMU in mode `model-exchange` (ME) 
fmu = fmiLoad("RobotRR", "Dymola", "2023x"; type = :ME)

# ╔═╡ c228eb10-d694-46aa-b952-01d824879287
begin

    # We activate the single instance mode, so only one FMU instance gets allocated and is reused again an again.
    fmiSingleInstanceMode!(fmu, true)

    using FMI.FMIImport: fmi2StringToValueReference

    # declare some model identifiers (inside of the FMU)
    STATE_I1 = fmu.modelDescription.stateValueReferences[2]
    STATE_I2 = fmu.modelDescription.stateValueReferences[1]
    STATE_A1 = fmi2StringToValueReference(
        fmu,
        "rRPositionControl_Elasticity.rr1.rotational1.revolute1.phi",
    )
    STATE_A2 = fmi2StringToValueReference(
        fmu,
        "rRPositionControl_Elasticity.rr1.rotational2.revolute1.phi",
    )
    STATE_dA1 = fmi2StringToValueReference(
        fmu,
        "rRPositionControl_Elasticity.rr1.rotational1.revolute1.w",
    )
    STATE_dA2 = fmi2StringToValueReference(
        fmu,
        "rRPositionControl_Elasticity.rr1.rotational2.revolute1.w",
    )

    DER_ddA2 = fmu.modelDescription.derivativeValueReferences[4]
    DER_ddA1 = fmu.modelDescription.derivativeValueReferences[6]

    VAR_TCP_PX = fmi2StringToValueReference(fmu, "rRPositionControl_Elasticity.tCP.p_x")
    VAR_TCP_PY = fmi2StringToValueReference(fmu, "rRPositionControl_Elasticity.tCP.p_y")
    VAR_TCP_VX = fmi2StringToValueReference(fmu, "rRPositionControl_Elasticity.tCP.v_x")
    VAR_TCP_VY = fmi2StringToValueReference(fmu, "rRPositionControl_Elasticity.tCP.v_y")
    VAR_TCP_F = fmi2StringToValueReference(fmu, "combiTimeTable.y[3]")

    HIDDEN_CODE_MESSAGE

end

# ╔═╡ 16ffc610-3c21-40f7-afca-e9da806ea626
md"""
Let's check out some meta data of the FMU with `fmiInfo`:
"""

# ╔═╡ 052f2f19-767b-4ede-b268-fce0aee133ad
fmiInfo(fmu)

# ╔═╡ 746fbf6f-ed7c-43b8-8a6f-0377cd3cf85e
md"""
> 👁️ We can read the model name, tool information for the exporting tool, number of event indicators, states, inputs, outputs and whether the optionally implemented FMI features (like *directional derivatives*) are supported by this FMU.
"""

# ╔═╡ 08e1ff54-d115-4da9-8ea7-5e89289723b3
md"""
All six states are listed with all their alias identifiers, that might look a bit awkward the first time. The six states - human readable - are:

| variable reference | description |
| -------- | ------ |
| 33554432 | motor #2 current |
| 33554433 | motor #1 current |
| 33554434 | joint #2 angle |
| 33554435 | joint #2 angular velocity |
| 33554436 | joint #1 angle |
| 33554437 | joint #1 angular velocity |
"""

# ╔═╡ 70c6b605-54fa-40a3-8bce-a88daf6a2022
md"""
To simulate - or *solve* - the ME-FMU, we need an ODE solver. We use the *Tsit5* (explicit Runge-Kutta) here.
"""

# ╔═╡ 634f923a-5e09-42c8-bac0-bf165ab3d12a
solver = Tsit5()

# ╔═╡ f59b5c84-2eae-4e3f-aaec-116c090d454d
md"""
Let's define an array of values we want to be recorded during the first simulation of our FMU. The variable identifiers (like `DER_ddA2`) were pre-defined in the hidden code section above.
"""

# ╔═╡ 0c9493c4-322e-41a0-9ec7-2e2c54ae1373
recordValues = [
    DER_ddA2,
    DER_ddA1, # mechanical accelerations
    STATE_A2,
    STATE_A1, # mechanical angles
    VAR_TCP_PX,
    VAR_TCP_PY, # tool-center-point x and y
    VAR_TCP_VX,
    VAR_TCP_VY, # tool-center-point velocity x and y
    VAR_TCP_F,
] # normal force pen on paper

# ╔═╡ 325c3032-4c78-4408-b86e-d9aa4cfc3187
md"""
Let's simulate the FMU using `fmiSimulate`. In the solution object, different information can be found, like the number of ODE, jacobian or gradient evaluations: 
"""

# ╔═╡ 25e55d1c-388f-469d-99e6-2683c0508693
sol_fmu_train = fmiSimulate(
    fmu,    # our FMU
    (tStart, tStop);           # sim. from tStart to tStop
    solver = solver,    # use the Tsit5 solver
    parameters = parameters,     # the word "train"
    saveat = tSave,    # saving points for the sol.
    recordValues = recordValues,
) # values to record

# ╔═╡ 74c519c9-0eef-4798-acff-b11044bb4bf1
md"""
Now that we know our model and data a little bit better, it's time to care about our hybrid model topology.

# Experiments: $br Hybrid Model Topology

Today is opposite day! Instead of deriving a topology step by step, the final neural FMU topology is presented in the picture below... however, three experiments are intended to make clear why it looks the way it looks.

![](https://github.com/ThummeTo/FMIFlux.jl/blob/main/examples/pluto-src/SciMLUsingFMUs/src/plan_complete.png?raw=true)

The first experiment is on choosing a good interface between FMU and ANN. The second is on online data pre- and post-processing. And the third one on gates, that allow to control the influence of ANN and FMU on the resulting hybrid model dynamics. After you completed all three, you are equipped with the knowledge to cope the final challenge: Build your own neural FMU and train it!
"""

# ╔═╡ 786c4652-583d-43e9-a101-e28c0b6f64e4
md"""
## Choosing interface signals
**between the physical and machine learning domain**

When connecting an FMU with an ANN, technically different signals could be used: States, state derivatives, inputs, outputs, parameters, time itself or other observable variables. Depending on the use case, some signals are more clever to choose than others. In general, every additional signal costs a little bit of computational performance, as you will see. So picking the right subset is the key!

![](https://github.com/ThummeTo/FMIFlux.jl/blob/main/examples/pluto-src/SciMLUsingFMUs/src/plan_e1.png?raw=true)
"""

# ╔═╡ 5d688c3d-b5e3-4a3a-9d91-0896cc001000
md"""
We start building our deep model as a `Chain` of layers. For now, there is only a single layer in it: The FMU `fmu` itself. The layer input `x` is interpreted as system state (compare to the figure above) and set in the fmu call via `x=x`. The current solver time `t` is set implicitly. Further, we want all state derivatives as layer outputs by setting `dx_refs=:all` and some additional outputs specified via `y_refs=CHOOSE_y_refs` (you can pick them using the checkboxes). 
"""

# ╔═╡ 68719de3-e11e-4909-99a3-5e05734cc8b1
md"""
Which signals are used for `y_refs`, can be selected:
"""

# ╔═╡ b42bf3d8-e70c-485c-89b3-158eb25d8b25
@bind CHOOSE_y_refs MultiCheckBox([
    STATE_A1 => "Angle Joint 1",
    STATE_A2 => "Angle Joint 2",
    STATE_dA1 => "Angular velocity Joint 1",
    STATE_dA2 => "Angular velocity Joint 2",
    VAR_TCP_PX => "TCP position x",
    VAR_TCP_PY => "TCP position y",
    VAR_TCP_VX => "TCP velocity x",
    VAR_TCP_VY => "TCP velocity y",
    VAR_TCP_F => "TCP (normal) force z",
])

# ╔═╡ 2e08df84-a468-4e99-a277-e2813dfeae5c
model = Chain(x -> fmu(; x = x, dx_refs = :all, y_refs = CHOOSE_y_refs))

# ╔═╡ c446ed22-3b23-487d-801e-c23742f81047
md"""
Let's pick a state `x1` one second after simulation start to determine sensitivities for:
"""

# ╔═╡ fc3d7989-ac10-4a82-8777-eeecd354a7d0
x1 = FMIZoo.getState(data_train, tStart + 1.0)

# ╔═╡ f4e66f76-76ff-4e21-b4b5-c1ecfd846329
begin
    using FMIFlux.FMISensitivity.ReverseDiff
    using FMIFlux.FMISensitivity.ForwardDiff

    prepareSolveFMU(fmu, parameters)
    jac_rwd = ReverseDiff.jacobian(x -> model(x), x1)
    A_rwd = jac_rwd[1:length(x1), :]
end

# ╔═╡ 0a7955e7-7c1a-4396-9613-f8583195c0a8
md"""
Depending on how many signals you select, the output of the FMU-layer is extended. The first six outputs are the state derivatives, the remaining are the $(length(CHOOSE_y_refs)) additional output(s) selected above.
"""

# ╔═╡ 4912d9c9-d68d-4afd-9961-5d8315884f75
begin
    dx_y = model(x1)
end

# ╔═╡ 19942162-cd4e-487c-8073-ea6b262d299d
md"""
Derivatives:
"""

# ╔═╡ 73575386-673b-40cc-b3cb-0b8b4f66a604
ẋ = dx_y[1:length(x1)]

# ╔═╡ 24861a50-2319-4c63-a800-a0a03279efe2
md"""
Additional outputs:
"""

# ╔═╡ 93735dca-c9f3-4f1a-b1bd-dfe312a0644a
y = dx_y[length(x1)+1:end]

# ╔═╡ 13ede3cd-99b1-4e65-8a18-9043db544728
md"""
For the later training, we need gradients and Jacobians.
"""

# ╔═╡ f7c119dd-c123-4c43-812e-d0625817d77e
md"""
If we use reverse-mode automatic differentiation via `ReverseDiff.jl`, the determined Jacobian $A = \frac{\partial \dot{x}}{\partial x}$ states: 
"""

# ╔═╡ b163115b-393d-4589-842d-03859f05be9a
md"""
Forward-mode automatic differentiation (using *ForwardDiff.jl*)is available, too.

We can determine further Jacobians for FMUs, for example the Jacobian $C = \frac{\partial y}{\partial x}$ states (using *ReverseDiff.jl*): 
"""

# ╔═╡ ac0afa6c-b6ec-4577-aeb6-10d1ec63fa41
begin
    C_rwd = jac_rwd[length(x1)+1:end, :]
end

# ╔═╡ 5e9cb956-d5ea-4462-a649-b133a77929b0
md"""
Let's check the performance of these calls, because they will have significant influence on the later training performance!
"""

# ╔═╡ 9dc93971-85b6-463b-bd17-43068d57de94
md"""
### Benchmark
The amount of selected signals has influence on the computational performance of the model. The more signals you use, the slower is inference and gradient determination. For now, you have picked $(length(CHOOSE_y_refs)) additional signal(s). 
"""

# ╔═╡ 476a1ed7-c865-4878-a948-da73d3c81070
begin
    CHOOSE_y_refs

    md"""
    🎬 **Start Benchmark** $(@bind BENCHMARK CheckBox())
    (benchmarking takes around 10 seconds)
    """
end

# ╔═╡ 0b6b4f6d-be09-42f3-bc2c-5f17a8a9ab0e
md"""
The current timing and allocations for inference are:
"""

# ╔═╡ a1aca180-d561-42a3-8d12-88f5a3721aae
begin
    if BENCHMARK
        @btime model(x1)
    else
        BENCHMARK_MESSAGE
    end
end

# ╔═╡ 3bc2b859-d7b1-4b79-88df-8fb517a6929d
md"""
Gradient and Jacobian computation takes a little longer of course. We use reverse-mode automatic differentiation via `ReverseDiff.jl` here:
"""

# ╔═╡ a501d998-6fd6-496f-9718-3340c42b08a6
begin
    if BENCHMARK
        prepareSolveFMU(fmu, parameters)
        function ben_rwd(x)
            return ReverseDiff.jacobian(model, x + rand(6) * 1e-12)
        end
        @btime ben_rwd(x1)
        #nothing
    else
        BENCHMARK_MESSAGE
    end
end

# ╔═╡ 83a2122d-56da-4a80-8c10-615a8f76c4c1
md"""
Further, forward-mode automatic differentiation is available too via `ForwardDiff.jl`, but a little bit slower than reverse-mode:
"""

# ╔═╡ e342be7e-0806-4f72-9e32-6d74ed3ed3f2
begin
    if BENCHMARK
        prepareSolveFMU(fmu, parameters)
        function ben_fwd(x)
            return ForwardDiff.jacobian(model, x + rand(6) * 1e-12)
        end
        @btime ben_fwd(x1) # second run for "benchmarking"
    #nothing
    else
        BENCHMARK_MESSAGE
    end
end

# ╔═╡ eaf37128-0377-42b6-aa81-58f0a815276b
md"""
> 💡 Keep in mind that the choice of interface might has a significant impact on your inference and training performance! However, some signals are simply required to be part of the interface, because the effect we want to train for depends on them.
"""

# ╔═╡ c030d85e-af69-49c9-a7c8-e490d4831324
md"""
## Online Data Pre- and Postprocessing
**is required for hybrid models**

Now that we have defined the signals that come *from* the FMU and go *into* the ANN, we need to think about data pre- and post-processing. In ML, this is often done before the actual training starts. In hybrid modeling, we need to do this *online*, because the FMU constantly generates signals that might not be suitable for ANNs. On the other hand, the signals generated by ANNs might not suit the expected FMU input. What *suitable* means gets more clear if we have a look on the used activation functions, like e.g. the *tanh*.

![](https://github.com/ThummeTo/FMIFlux.jl/blob/main/examples/pluto-src/SciMLUsingFMUs/src/plan_e2.png?raw=true)

We simplify the ANN to a single nonlinear activation function. Let's see what's happening as soon as we put the derivative *angular velocity of joint 1* (dα1) from the FMU into a `tanh` function:
"""

# ╔═╡ 51c200c9-0de3-4e50-8884-49fe06158560
begin
    fig_pre_post1 = plot(
        layout = grid(1, 2, widths = (1 / 4, 3 / 4)),
        xlabel = "t [s]",
        legend = :bottomright,
    )

    plot!(fig_pre_post1[1], data_train.t, data_train.da1, label = :none, xlims = (0.0, 0.1))
    plot!(fig_pre_post1[1], data_train.t, tanh.(data_train.da1), label = :none)

    plot!(fig_pre_post1[2], data_train.t, data_train.da1, label = "dα1")
    plot!(fig_pre_post1[2], data_train.t, tanh.(data_train.da1), label = "tanh(dα1)")

    fig_pre_post1
end

# ╔═╡ 0dadd112-3132-4491-9f02-f43cf00aa1f9
md"""
In general, it looks like the velocity isn't saturated too much by `tanh`. This is a good thing and not always the case! However, the very beginning of the trajectory is saturated too much (the peak value of $\approx -3$ is saturated to $\approx -1$). This is bad, because the hybrid model velocity is *slower* in this time interval and the hybrid system won't reach the same angle over time as the original FMU.

We can add shift (=addition) and scale (=multiplication) operations before and after the ANN to bypass this issue. See how you can influence the output *after* the `tanh` (and the ANN respectively) to match the ranges. The goal is to choose pre- and post-processing parameters so that the signal ranges needed by the FMU are preserved by the hybrid model.
"""

# ╔═╡ bf6bf640-54bc-44ef-bd4d-b98e934d416e
@bind PRE_POST_SHIFT Slider(-1:0.1:1.0, default = 0.0)

# ╔═╡ 5c2308d9-6d04-4b38-af3b-6241da3b6871
md"""
Change the `shift` value $(PRE_POST_SHIFT):
"""

# ╔═╡ 007d6d95-ad85-4804-9651-9ac3703d3b40
@bind PRE_POST_SCALE Slider(0.1:0.1:2.0, default = 1.0)

# ╔═╡ 639889b3-b9f2-4a3c-999d-332851768fd7
md"""
Change the `scale` value $(PRE_POST_SCALE):
"""

# ╔═╡ ed1887df-5079-4367-ab04-9d02a1d6f366
begin
    fun_pre = ShiftScale([PRE_POST_SHIFT], [PRE_POST_SCALE])
    fun_post = ScaleShift(fun_pre)

    fig_pre_post2 = plot(; layout = grid(1, 2, widths = (1 / 4, 3 / 4)), xlabel = "t [s]")

    plot!(
        fig_pre_post2[2],
        data_train.t,
        data_train.da1,
        label = :none,
        title = "Shift: $(round(PRE_POST_SHIFT; digits=1)) | Scale: $(round(PRE_POST_SCALE; digits=1))",
        legend = :bottomright,
    )
    plot!(fig_pre_post2[2], data_train.t, tanh.(data_train.da1), label = :none)
    plot!(
        fig_pre_post2[2],
        data_train.t,
        fun_post(tanh.(fun_pre(data_train.da1))),
        label = :none,
    )

    plot!(fig_pre_post2[1], data_train.t, data_train.da1, label = "dα1", xlims = (0.0, 0.1))
    plot!(fig_pre_post2[1], data_train.t, tanh.(data_train.da1), label = "tanh(dα1)")
    plot!(
        fig_pre_post2[1],
        data_train.t,
        fun_post(tanh.(fun_pre(data_train.da1))),
        label = "post(tanh(pre(dα1)))",
    )

    fig_pre_post2
end

# ╔═╡ 0b0c4650-2ce1-4879-9acd-81c16d06700e
md"""
The left plot shows the negative spike at the very beginning in more detail. In *FMIFlux.jl*, there are ready to use layers for scaling and shifting, that can automatically select appropriate parameters. These parameters are trained together with the ANN parameters by default, so they can adapt to new signal ranges that might occur during training.
"""

# ╔═╡ b864631b-a9f3-40d4-a6a8-0b57a37a476d
md"""
> 💡 In many machine learning applications, pre- and post-processing is done offline. If we combine machine learning and physical models, we need to pre- and post-process online at the interfaces. This does at least improve training performance and is a necessity if the nominal values become very large or very small.
"""

# ╔═╡ 0fb90681-5d04-471a-a7a8-4d0f3ded7bcf
md"""
## Introducing Gates
**to control how physical and machine learning model contribute and interact**

![](https://github.com/ThummeTo/FMIFlux.jl/blob/main/examples/pluto-src/SciMLUsingFMUs/src/plan_e3.png?raw=true)
"""

# ╔═╡ 95e14ea5-d82d-4044-8c68-090d74d95a61
md"""
There are two obvious ways of connecting two blocks (the ANN and the FMU):
- In **series**, so one block is getting signals from the other block and is able to *manipulate* or *correct* these signals. This way, e.g. modeling or parameterization errors can be corrected.
- In **parallel**, so both are getting the same signals and calculate own outputs, these outputs must be merged afterwards. This way, additional system parts, like e.g. forces or momentum, can be learned and added to or augment the existing dynamics.

The good news is, you don't have to decide this beforehand. This is something that the optimizer can decide, if we introduce a topology with parameters, that allow for both modes. This structure is referred to as *gates*.
"""

# ╔═╡ cbae6aa4-1338-428c-86aa-61d3304e33ed
@bind GATE_INIT_FMU Slider(0.0:0.1:1.0, default = 1.0)

# ╔═╡ 2fa1821b-aaec-4de4-bfb4-89560790dc39
md"""
Change the opening of the **FMU gate** $(GATE_INIT_FMU) for dα1:
"""

# ╔═╡ 8c56acd6-94d3-4cbc-bc29-d249740268a0
@bind GATE_INIT_ANN Slider(0.0:0.1:1.0, default = 0.0)

# ╔═╡ 9b52a65a-f20c-4387-aaca-5292a92fb639
md"""
Change the opening of the **ANN gate** $(GATE_INIT_ANN) for dα1:
"""

# ╔═╡ 845a95c4-9a35-44ae-854c-57432200da1a
md"""
The FMU gate value for dα1 is $(GATE_INIT_FMU) and the ANN gate value is $(GATE_INIT_ANN). This means the hybrid model dα1 is composed of $(GATE_INIT_FMU*100)% of dα1 from the FMU and of $(GATE_INIT_ANN*100)% of dα1 from the ANN.
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
        # the accelerations [1,3] and consumptions [2,4] are paired
        gates = ScaleSum([GATE_INIT_FMU, GATE_INIT_ANN], [[1, 2]]) # gates with sum

        # setup the neural FMU topology
        model_gates = Flux.f64(
            Chain(
                dx -> cache(dx),                    # cache `dx`
                Dense(1, 16, tanh),
                Dense(16, 1, tanh),  # pre-process `dx`
                dx -> cacheRetrieve(1, dx),       # dynamics FMU | dynamics ANN
                gates,
            ),
        )       # stack together

        model_input = collect([v] for v in data_train.da1)
        model_output = collect(model_gates(inp) for inp in model_input)
        ANN_output = collect(model_gates[2:3](inp) for inp in model_input)

        fig = plot(; ylims = (-3, 1), legend = :bottomright)
        plot!(fig, data_train.t, collect(v[1] for v in model_input), label = "dα1 of FMU")
        plot!(fig, data_train.t, collect(v[1] for v in ANN_output), label = "dα1 of ANN")
        plot!(
            fig,
            data_train.t,
            collect(v[1] for v in model_output),
            label = "dα1 of neural FMU",
        )

        return fig
    end
    build_model_gates()
end

# ╔═╡ fd1cebf1-5ccc-4bc5-99d4-1eaa30e9762e
md"""
Some observations from the current gate openings are:

This equals the serial topology: $((GATE_INIT_FMU==0 && GATE_INIT_ANN==1) ? "✔️" : "❌") $br
This equals the parallel topology: $((GATE_INIT_FMU==1 && GATE_INIT_ANN==1) ? "✔️" : "❌") $br
The neural FMU dynamics equal the FMU dynamics: $((GATE_INIT_FMU==1 && GATE_INIT_ANN==0) ? "✔️" : "❌")
"""

# ╔═╡ 1cd976fb-db40-4ebe-b40d-b996e16fc213
md"""
> 💡 Gates allow to make parts of the architecture *learnable* while still keeping the training results interpretable.
"""

# ╔═╡ 93771b35-4edd-49e3-bed1-a3ccdb7975e6
md"""
> 💭 **Further reading:** Optimizing the gates together with the ANN parameters seems a useful strategy if we don't know how FMU and ANN need to interact in the later application. Technically, we keep a part of the architecture *parameterizable* and therefore learnable. How far can we push this game?
>
> Actually to the point, that the combination of FMU and ANN is described by a single *connection* equation, that is able to express all possible combinations of both models with each other - so a connection between every pair of inputs and outputs. This is discussed in detail as part of our article [*Learnable & Interpretable Model Combination in Dynamic Systems Modeling*](https://doi.org/10.48550/arXiv.2406.08093).
"""

# ╔═╡ e79badcd-0396-4a44-9318-8c6b0a94c5c8
md"""
Time to take care of the big picture next.
"""

# ╔═╡ 2a5157c5-f5a2-4330-b2a3-0c1ec0b7adff
md"""
# Building the neural FMU
**... putting everything together**

![](https://github.com/ThummeTo/FMIFlux.jl/blob/main/examples/pluto-src/SciMLUsingFMUs/src/plan_train.png?raw=true)
"""

# ╔═╡ 4454c8d2-68ed-44b4-adfa-432297cdc957
md"""
## FMU inputs
In general, you can use arbitrary values as input for the FMU layer, like system inputs, states or parameters. In this example, we want to use only system states as inputs for the FMU layer - to keep it easy - which are:
- currents of both motors
- angles of both joints
- angular velocities of both joints

To preserve the ODE topology (a mapping from state to state derivative), we use all system state derivatives as layer outputs. However, you can choose further outputs if you want to... and you definitely should.

## ANN inputs
As input to the ANN, we choose at least the angular accelerations of both joints - this is fixed:

- angular acceleration Joint 1
- angular acceleration Joint 2

Pick additional ANN layer inputs:
"""

# ╔═╡ d240c95c-5aba-4b47-ab8d-2f9c0eb854cd
@bind y_refs MultiCheckBox([
    STATE_A2 => "Angle Joint 2",
    STATE_A1 => "Angle Joint 1",
    STATE_dA1 => "Angular velocity Joint 1",
    STATE_dA2 => "Angular velocity Joint 2",
    VAR_TCP_PX => "TCP position x",
    VAR_TCP_PY => "TCP position y",
    VAR_TCP_VX => "TCP velocity x",
    VAR_TCP_VY => "TCP velocity y",
    VAR_TCP_F => "TCP (normal) force z",
])

# ╔═╡ 06937575-9ab1-41cd-960c-7eef3e8cae7f
md"""
It might be clever to pick additional inputs, because the effect being learned (slip-stick of the pen) might depend on these additional inputs. However, every additional signal has a little negative impact on the computational performance and a risk of learning from wrong correlations.
"""

# ╔═╡ 356b6029-de66-418f-8273-6db6464f9fbf
md"""
## ANN size
"""

# ╔═╡ 5805a216-2536-44ac-a702-d92e86d435a4
md"""
The ANN shall have $(@bind NUM_LAYERS Select([2, 3, 4])) layers with a width of $(@bind LAYERS_WIDTH Select([8, 16, 32])) each.
"""

# ╔═╡ 53e971d8-bf43-41cc-ac2b-20dceaa78667
@bind GATES_INIT Slider(0.0:0.1:1.0, default = 0.0)

# ╔═╡ 68d57a23-68c3-418c-9c6f-32bdf8cafceb
md"""
The ANN gates shall be initialized with $(GATES_INIT), slide to change:
"""

# ╔═╡ e8b8c63b-2ca4-4e6a-a801-852d6149283e
md"""
ANN gates shall be initialized with $(GATES_INIT), meaning the ANN contributes $(GATES_INIT*100)% to the hybrid model derivatives, while the FMU contributes $(100-GATES_INIT*100)%. These parameters are adapted during training, these are only start values.
"""

# ╔═╡ c0ac7902-0716-4f18-9447-d18ce9081ba5
md"""
## Resulting neural FMU
Our final neural FMU topology looks like this:
"""

# ╔═╡ 84215a73-1ab0-416d-a9db-6b29cd4f5d2a
begin

    function build_topology(gates_init, add_y_refs, nl, lw)

        ANN_input_Vars = [recordValues[1:2]..., add_y_refs...]
        ANN_input_Vals = fmiGetSolutionValue(sol_fmu_train, ANN_input_Vars)
        ANN_input_Idcs = [4, 6]
        for i = 1:length(add_y_refs)
            push!(ANN_input_Idcs, i + 6)
        end

        # pre- and post-processing
        preProcess = ShiftScale(ANN_input_Vals)         # we put in the derivatives recorded above, FMIFlux shift and scales so we have a data mean of 0 and a standard deviation of 1
        #preProcess.scale[:] *= 0.1                         # add some additional "buffer"
        postProcess = ScaleShift(preProcess; indices = [1, 2])   # initialize the postProcess as inverse of the preProcess, but only take indices 1 and 2

        # cache
        cache = CacheLayer()                        # allocate a cache layer
        cacheRetrieve = CacheRetrieveLayer(cache)   # allocate a cache retrieve layer, link it to the cache layer

        gates = ScaleSum(
            [1.0 - gates_init, 1.0 - gates_init, gates_init, gates_init],
            [[1, 3], [2, 4]],
        ) # gates with sum

        ANN_layers = []
        push!(ANN_layers, Dense(2 + length(add_y_refs), lw, tanh)) # first layer 
        for i = 3:nl
            push!(ANN_layers, Dense(lw, lw, tanh))
        end
        push!(ANN_layers, Dense(lw, 2, tanh)) # last layer 

        model = Flux.f64(
            Chain(
                x -> fmu(; x = x, dx_refs = :all, y_refs = add_y_refs),
                dxy -> cache(dxy),                    # cache `dx`
                dxy -> dxy[ANN_input_Idcs],
                preProcess,
                ANN_layers...,
                postProcess,
                dx -> cacheRetrieve(4, 6, dx),       # dynamics FMU | dynamics ANN
                gates,                              # compute resulting dx from ANN + FMU
                dx -> cacheRetrieve(1:3, dx[1], 5, dx[2]),
            ),
        )

        return model

    end

    HIDDEN_CODE_MESSAGE

end

# ╔═╡ bc09bd09-2874-431a-bbbb-3d53c632be39
md"""
We find a `Chain` consisting of multipl layers and the corresponding parameter counts. We can evaluate it, by putting in our start state `x0`. The model computes the resulting state derivative:
"""

# ╔═╡ f02b9118-3fb5-4846-8c08-7e9bbca9d208
md"""
On basis of this `Chain`, we can build a neural FMU very easy:
"""

# ╔═╡ d347d51b-743f-4fec-bed7-6cca2b17bacb
md"""
So let's get that thing trained!

# Training

After setting everything up, we can give it a try and train our created neural FMU. Depending on the chosen optimization hyperparameters, this will be more or less successful. Feel free to play around a bit, but keep in mind that for real application design, you should do hyper parameter optimization instead of playing around by yourself.
"""

# ╔═╡ d60d2561-51a4-4f8a-9819-898d70596e0c
md"""
## Hyperparameters
Besides the already introduced hyperparameters - the depth, width and initial gate opening of the hybrid model - further parameters might have significant impact on the training success.

### Optimizer
For this example, we use the well-known `Adam`-Optimizer with a step size `eta` of $(@bind ETA Select([1e-4 => "1e-4", 1e-3 => "1e-3", 1e-2 => "1e-2"])). 

### Batching 
Because data has a significant length, gradient computation over the entire simulation trajectory might not be effective. The most common approach is to *cut* data into slices and train on these subsets instead of the entire trajectory at once. In this example, data is cut in pieces with length of $(@bind BATCHDUR Select([0.05, 0.1, 0.15, 0.2])) seconds.
"""

# ╔═╡ c97f2dea-cb18-409d-9ae8-1d03647a6bb3
md"""
This results in a batch with $(round(Integer, data_train.t[end] / BATCHDUR)) elements.
"""

# ╔═╡ 366abd1a-bcb5-480d-b1fb-7c76930dc8fc
md"""
We use a simple `Random` scheduler here, that picks a random batch element for the next training step. Other schedulers are pre-implemented in *FMIFlux.jl*.
"""

# ╔═╡ 7e2ffd6f-19b0-435d-8e3c-df24a591bc55
md"""
### Loss Function
Different loss functions are thinkable here. Two quantities that should be considered are the motor currents and the motor revolution speeds. For this workshop we use the *Mean Average Error* (MAE) over the motor currents. Other loss functions can easily be deployed.
"""

# ╔═╡ caa5e04a-2375-4c56-8072-52c140adcbbb
# goal is to match the motor currents (they can be recorded easily in the real application)
function loss(solution::FMU2Solution, data::FMIZoo.RobotRR_Data)

    # determine the start/end indices `ts` and `te` (sampled with 100Hz)
    dt = 0.01
    ts = 1 + round(Integer, solution.states.t[1] / dt)
    te = 1 + round(Integer, solution.states.t[end] / dt)

    # retrieve simulation data from neural FMU ("where we are") and data from measurements ("where we want to be")
    i1_value = fmiGetSolutionState(solution, STATE_I1)
    i2_value = fmiGetSolutionState(solution, STATE_I2)
    i1_data = data.i1[ts:te]
    i2_data = data.i2[ts:te]

    # accumulate our loss value
    Δvalue = 0.0
    Δvalue += FMIFlux.Losses.mae(i1_value, i1_data)
    Δvalue += FMIFlux.Losses.mae(i2_value, i2_data)

    return Δvalue
end

# ╔═╡ 69657be6-6315-4655-81e2-8edef7f21e49
md"""
For example, the loss function value of the plain FMU is $(round(loss(sol_fmu_train, data_train); digits=6)).
"""

# ╔═╡ 23ad65c8-5723-4858-9abe-750c3b65c28a
md"""
## Summary
To summarize, your ANN has a **depth of $(NUM_LAYERS) layers** with a **width of $(LAYERS_WIDTH)** each. The **ANN gates are initialized with $(GATES_INIT*100)%**, so all FMU gates are initialized with $(100-GATES_INIT*100)%. You decided to batch your data with a **batch element length of $(BATCHDUR)** seconds. Besides the state derivatives, you **put $(length(y_refs)) additional variables** in the ANN. Adam optimizer will try to find a good minimum with **`eta` is $(ETA)**.

Batching takes a few seconds and training a few minutes (depending on the number of training steps), so this is not triggered automatically. If you are ready to go, choose a number of training steps and check the checkbox `Start Training`. This will start a training of $(@bind STEPS Select([0, 10, 100, 1000, 2500, 5000, 10000])) training steps. Alternatively, you can change the training mode to `demo` which loads parameters from a pre-trained model.
"""

# ╔═╡ abc57328-4de8-42d8-9e79-dd4020769dd9
md"""
Select training mode:
$(@bind MODE Select([:train => "Training", :demo => "Demo (pre-trained)"]))
"""

# ╔═╡ f9d35cfd-4ae5-4dcd-94d9-02aefc99bdfb
begin
    using JLD2

    if MODE == :train
        final_model = build_topology(GATES_INIT, y_refs, NUM_LAYERS, LAYERS_WIDTH)
    elseif MODE == :demo
        final_model = build_topology(
            0.2,
            [STATE_A2, STATE_A1, VAR_TCP_VX, VAR_TCP_VY, VAR_TCP_F],
            3,
            32,
        )
    end
end

# ╔═╡ f741b213-a20d-423a-a382-75cae1123f2c
final_model(x0)

# ╔═╡ 91473bef-bc23-43ed-9989-34e62166d455
begin
    neuralFMU = ME_NeuralFMU(
        fmu, # the FMU used in the neural FMU 
        final_model,    # the model we specified above 
        (tStart, tStop),# start and stop time for solving
        solver; # the solver (Tsit5)
        saveat = tSave,
    )   # time points to save the solution at
end

# ╔═╡ 404ca10f-d944-4a9f-addb-05efebb4f159
begin
    import Downloads
    demo_path = Downloads.download(
        "https://github.com/ThummeTo/FMIFlux.jl/blob/main/examples/pluto-src/SciMLUsingFMUs/src/20000.jld2?raw=true",
    )

    # in demo mode, we load parameters from a pre-trained model
    if MODE == :demo
        fmiLoadParameters(neuralFMU, demo_path)
    end

    HIDDEN_CODE_MESSAGE
end

# ╔═╡ e8bae97d-9f90-47d2-9263-dc8fc065c3d0
begin
    neuralFMU
    y_refs
    NUM_LAYERS
    LAYERS_WIDTH
    GATES_INIT
    ETA
    BATCHDUR
    MODE

    if MODE == :train
        md"""⚠️ The roughly estimated training time is **$(round(Integer, STEPS*10*BATCHDUR + 0.6/BATCHDUR)) seconds** (Windows, i7 @ 3.6GHz). Training might be faster if the system is less stiff than expected. Once you started training by clicking on `Start Training`, training can't be terminated easily.
      	
      🎬 **Start Training** $(@bind LIVE_TRAIN CheckBox())
      		"""
    else
        LIVE_TRAIN = false
        md"""ℹ️ No training in demo mode. Please continue with plotting results.
        """
    end
end

# ╔═╡ 2dce68a7-27ec-4ffc-afba-87af4f1cb630
begin

    function train(eta, batchdur, steps)

        if steps == 0
            return md"""⚠️ Number of training steps is `0`, no training."""
        end

        prepareSolveFMU(fmu, parameters)

        train_t = data_train.t
        train_data = collect([data_train.i2[i], data_train.i1[i]] for i = 1:length(train_t))

        #@info 
        @info "Started batching ..."

        batch = batchDataSolution(
            neuralFMU, # our neural FMU model
            t -> FMIZoo.getState(data_train, t), # a function returning a start state for a given time point `t`, to determine start states for batch elements
            train_t, # data time points
            train_data; # data cumulative consumption 
            batchDuration = batchdur, # duration of one batch element
            indicesModel = [1, 2], # model indices to train on (1 and 2 equal the `electrical current` states)
            plot = false, # don't show intermediate plots (try this outside of Pluto)
            showProgress = false,
            parameters = parameters,
        )

        @info "... batching finished!"

        # a random element scheduler
        scheduler = RandomScheduler(neuralFMU, batch; applyStep = 1, plotStep = 0)

        lossFct = (solution::FMU2Solution) -> loss(solution, data_train)

        maxiters = round(Int, 1e5 * batchdur)

        _loss =
            p -> FMIFlux.Losses.loss(
                neuralFMU, # the neural FMU to simulate
                batch; # the batch to take an element from
                p = p, # the neural FMU training parameters (given as input)
                lossFct = lossFct, # our custom loss function
                batchIndex = scheduler.elementIndex, # the index of the batch element to take, determined by the chosen scheduler
                logLoss = true, # log losses after every evaluation
                showProgress = false,
                parameters = parameters,
                maxiters = maxiters,
            )

        params = FMIFlux.params(neuralFMU)

        FMIFlux.initialize!(
            scheduler;
            p = params[1],
            showProgress = false,
            parameters = parameters,
            print = false,
        )

        BETA1 = 0.9
        BETA2 = 0.999
        optim = Adam(eta, (BETA1, BETA2))

        @info "Started training ..."

        @withprogress name = "iterating" begin
            iteration = 0
            function cb()
                iteration += 1
                @logprogress iteration / steps
                FMIFlux.update!(scheduler; print = false)
                nothing
            end

            FMIFlux.train!(
                _loss, # the loss function for training
                neuralFMU, # the parameters to train
                Iterators.repeated((), steps), # an iterator repeating `steps` times
                optim; # the optimizer to train
                gradient = :ReverseDiff, # use ReverseDiff, because it's much faster!
                cb = cb, # update the scheduler after every step 
                proceed_on_assert = true,
            ) # go on if a training steps fails (e.g. because of instability)  
        end

        @info "... training finished!"
    end

    HIDDEN_CODE_MESSAGE

end

# ╔═╡ c3f5704b-8e98-4c46-be7a-18ab4f139458
let
    if MODE == :train
        if LIVE_TRAIN
            train(ETA, BATCHDUR, STEPS)
        else
            LIVE_TRAIN_MESSAGE
        end
    else
        md"""ℹ️ No training in demo mode. Please continue with plotting results.
        """
    end
end

# ╔═╡ 1a608bc8-7264-4dd3-a4e7-0e39128a8375
md"""
> 💡 Playing around with hyperparameters is fun, but keep in mind that this is not a suitable method for finding good hyperparameters in real world engineering. Do a hyperparameter optimization instead.
"""

# ╔═╡ ff106912-d18c-487f-bcdd-7b7af2112cab
md"""
# Results 
Now it's time to find out if it worked! Plotting results makes the notebook slow, so it's deactivated by default. Activate it to plot results of your training.

## Training results
Let's check out the *training* results of the freshly trained neural FMU.
"""

# ╔═╡ 51eeb67f-a984-486a-ab8a-a2541966fa72
begin
    neuralFMU
    MODE
    LIVE_TRAIN
    md"""
    🎬 **Plot results** $(@bind LIVE_RESULTS CheckBox()) 
    """
end

# ╔═╡ 27458e32-5891-4afc-af8e-7afdf7e81cc6
begin

    function plotPaths!(fig, t, x, N; color = :black, label = :none, kwargs...)
        paths = []
        path = nothing
        lastN = N[1]
        for i = 1:length(N)
            if N[i] == 0.0
                if lastN == 1.0
                    push!(path, (t[i], x[i]))
                    push!(paths, path)
                end
            end

            if N[i] == 1.0
                if lastN == 0.0
                    path = []
                end
                push!(path, (t[i], x[i]))
            end

            lastN = N[i]
        end
        if length(path) > 0
            push!(paths, path)
        end

        isfirst = true
        for path in paths
            plot!(
                fig,
                collect(v[1] for v in path),
                collect(v[2] for v in path);
                label = isfirst ? label : :none,
                color = color,
                kwargs...,
            )
            isfirst = false
        end

        return fig
    end

    HIDDEN_CODE_MESSAGE

end

# ╔═╡ 737e2c50-0858-4205-bef3-f541e33b85c3
md"""
### FMU
Simulating the FMU (training data):
"""

# ╔═╡ 5dd491a4-a8cd-4baf-96f7-7a0b850bb26c
begin
    fmu_train = fmiSimulate(
        fmu,
        (data_train.t[1], data_train.t[end]);
        x0 = x0,
        parameters = Dict{String,Any}("fileName" => data_train.params["fileName"]),
        recordValues = [
            "rRPositionControl_Elasticity.tCP.p_x",
            "rRPositionControl_Elasticity.tCP.p_y",
            "rRPositionControl_Elasticity.tCP.N",
            "rRPositionControl_Elasticity.tCP.a_x",
            "rRPositionControl_Elasticity.tCP.a_y",
        ],
        showProgress = true,
        maxiters = 1e6,
        saveat = data_train.t,
        solver = Tsit5(),
    )
    nothing
end

# ╔═╡ 4f27b6c0-21da-4e26-aaad-ff453c8af3da
md"""
### Neural FMU
Simulating the neural FMU (training data):
"""

# ╔═╡ 1195a30c-3b48-4bd2-8a3a-f4f74f3cd864
begin
    if LIVE_RESULTS
        result_train = neuralFMU(
            x0,
            (data_train.t[1], data_train.t[end]);
            parameters = Dict{String,Any}("fileName" => data_train.params["fileName"]),
            recordValues = [
                "rRPositionControl_Elasticity.tCP.p_x",
                "rRPositionControl_Elasticity.tCP.p_y",
                "rRPositionControl_Elasticity.tCP.N",
                "rRPositionControl_Elasticity.tCP.v_x",
                "rRPositionControl_Elasticity.tCP.v_y",
            ],
            showProgress = true,
            maxiters = 1e6,
            saveat = data_train.t,
        )
        nothing
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ b0ce7b92-93e0-4715-8324-3bf4ff42a0b3
let
    if LIVE_RESULTS
        loss_fmu = loss(fmu_train, data_train)
        loss_nfmu = loss(result_train, data_train)

        md"""
      #### The word `train`
      The loss function value of the FMU on training data is $(round(loss_fmu; digits=6)), of the neural FMU it is $(round(loss_nfmu; digits=6)). The neural FMU is about $(round(loss_fmu/loss_nfmu; digits=1)) times more accurate.
      """
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ 919419fe-35de-44bb-89e4-8f8688bee962
let
    if LIVE_RESULTS
        fig = plot(; dpi = 300, size = (200 * 3, 60 * 3))
        plotPaths!(
            fig,
            data_train.tcp_px,
            data_train.tcp_py,
            data_train.tcp_norm_f,
            label = "Data",
            color = :black,
            style = :dash,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in fmu_train.values.saveval),
            collect(v[2] for v in fmu_train.values.saveval),
            collect(v[3] for v in fmu_train.values.saveval),
            label = "FMU",
            color = :orange,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in result_train.values.saveval),
            collect(v[2] for v in result_train.values.saveval),
            collect(v[3] for v in result_train.values.saveval),
            label = "Neural FMU",
            color = :blue,
        )
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ ed25a535-ca2f-4cd2-b0af-188e9699f1c3
md"""
#### The letter `a`
"""

# ╔═╡ 2918daf2-6499-4019-a04b-8c3419ee1ab7
let
    if LIVE_RESULTS
        fig = plot(;
            dpi = 300,
            size = (40 * 10, 40 * 10),
            xlims = (0.165, 0.205),
            ylims = (-0.035, 0.005),
        )
        plotPaths!(
            fig,
            data_train.tcp_px,
            data_train.tcp_py,
            data_train.tcp_norm_f,
            label = "Data",
            color = :black,
            style = :dash,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in fmu_train.values.saveval),
            collect(v[2] for v in fmu_train.values.saveval),
            collect(v[3] for v in fmu_train.values.saveval),
            label = "FMU",
            color = :orange,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in result_train.values.saveval),
            collect(v[2] for v in result_train.values.saveval),
            collect(v[3] for v in result_train.values.saveval),
            label = "Neural FMU",
            color = :blue,
        )
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ d798a5d0-3017-4eab-9cdf-ee85d63dfc49
md"""
#### The letter `n`
"""

# ╔═╡ 048e39c3-a3d9-4e6b-b050-1fd5a919e4ae
let
    if LIVE_RESULTS
        fig = plot(;
            dpi = 300,
            size = (50 * 10, 40 * 10),
            xlims = (0.245, 0.295),
            ylims = (-0.04, 0.0),
        )
        plotPaths!(
            fig,
            data_train.tcp_px,
            data_train.tcp_py,
            data_train.tcp_norm_f,
            label = "Data",
            color = :black,
            style = :dash,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in fmu_train.values.saveval),
            collect(v[2] for v in fmu_train.values.saveval),
            collect(v[3] for v in fmu_train.values.saveval),
            label = "FMU",
            color = :orange,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in result_train.values.saveval),
            collect(v[2] for v in result_train.values.saveval),
            collect(v[3] for v in result_train.values.saveval),
            label = "Neural FMU",
            color = :blue,
        )
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ b489f97d-ee90-48c0-af06-93b66a1f6d2e
md"""
## Validation results
Let's check out the *validation* results of the freshly trained neural FMU.
"""

# ╔═╡ 4dad3e55-5bfd-4315-bb5a-2680e5cbd11c
md"""
### FMU
Simulating the FMU (validation data):
"""

# ╔═╡ ea0ede8d-7c2c-4e72-9c96-3260dc8d817d
begin
    fmu_validation = fmiSimulate(
        fmu,
        (data_validation.t[1], data_validation.t[end]);
        x0 = x0,
        parameters = Dict{String,Any}("fileName" => data_validation.params["fileName"]),
        recordValues = [
            "rRPositionControl_Elasticity.tCP.p_x",
            "rRPositionControl_Elasticity.tCP.p_y",
            "rRPositionControl_Elasticity.tCP.N",
        ],
        showProgress = true,
        maxiters = 1e6,
        saveat = data_validation.t,
        solver = Tsit5(),
    )
    nothing
end

# ╔═╡ 35f52dbc-0c0b-495e-8fd4-6edbc6fa811e
md"""
### Neural FMU
Simulating the neural FMU (validation data):
"""

# ╔═╡ 51aed933-2067-4ea8-9c2f-9d070692ecfc
begin
    if LIVE_RESULTS
        result_validation = neuralFMU(
            x0,
            (data_validation.t[1], data_validation.t[end]);
            parameters = Dict{String,Any}("fileName" => data_validation.params["fileName"]),
            recordValues = [
                "rRPositionControl_Elasticity.tCP.p_x",
                "rRPositionControl_Elasticity.tCP.p_y",
                "rRPositionControl_Elasticity.tCP.N",
            ],
            showProgress = true,
            maxiters = 1e6,
            saveat = data_validation.t,
        )
        nothing
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ 8d9dc86e-f38b-41b1-80c6-b2ab6f488a3a
begin
    if LIVE_RESULTS
        loss_fmu = loss(fmu_validation, data_validation)
        loss_nfmu = loss(result_validation, data_validation)
        md"""
      #### The word `validate`
      The loss function value of the FMU on validation data is $(round(loss_fmu; digits=6)), of the neural FMU it is $(round(loss_nfmu; digits=6)). The neural FMU is about $(round(loss_fmu/loss_nfmu; digits=1)) times more accurate.
      """
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ 74ef5a39-1dd7-404a-8baf-caa1021d3054
let
    if LIVE_RESULTS
        fig = plot(; dpi = 300, size = (200 * 3, 40 * 3))
        plotPaths!(
            fig,
            data_validation.tcp_px,
            data_validation.tcp_py,
            data_validation.tcp_norm_f,
            label = "Data",
            color = :black,
            style = :dash,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in fmu_validation.values.saveval),
            collect(v[2] for v in fmu_validation.values.saveval),
            collect(v[3] for v in fmu_validation.values.saveval),
            label = "FMU",
            color = :orange,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in result_validation.values.saveval),
            collect(v[2] for v in result_validation.values.saveval),
            collect(v[3] for v in result_validation.values.saveval),
            label = "Neural FMU",
            color = :blue,
        )
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ 347d209b-9d41-48b0-bee6-0d159caacfa9
md"""
#### The letter `d`
"""

# ╔═╡ 05281c4f-dba8-4070-bce3-dc2f1319902e
let
    if LIVE_RESULTS
        fig = plot(;
            dpi = 300,
            size = (35 * 10, 50 * 10),
            xlims = (0.188, 0.223),
            ylims = (-0.025, 0.025),
        )
        plotPaths!(
            fig,
            data_validation.tcp_px,
            data_validation.tcp_py,
            data_validation.tcp_norm_f,
            label = "Data",
            color = :black,
            style = :dash,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in fmu_validation.values.saveval),
            collect(v[2] for v in fmu_validation.values.saveval),
            collect(v[3] for v in fmu_validation.values.saveval),
            label = "FMU",
            color = :orange,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in result_validation.values.saveval),
            collect(v[2] for v in result_validation.values.saveval),
            collect(v[3] for v in result_validation.values.saveval),
            label = "Neural FMU",
            color = :blue,
        )
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ 590d7f24-c6b6-4524-b3db-0c93d9963b74
md"""
#### The letter `t`
"""

# ╔═╡ 67cfe7c5-8e62-4bf0-996b-19597d5ad5ef
let
    if LIVE_RESULTS
        fig = plot(;
            dpi = 300,
            size = (25 * 10, 50 * 10),
            xlims = (0.245, 0.27),
            ylims = (-0.025, 0.025),
            legend = :topleft,
        )
        plotPaths!(
            fig,
            data_validation.tcp_px,
            data_validation.tcp_py,
            data_validation.tcp_norm_f,
            label = "Data",
            color = :black,
            style = :dash,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in fmu_validation.values.saveval),
            collect(v[2] for v in fmu_validation.values.saveval),
            collect(v[3] for v in fmu_validation.values.saveval),
            label = "FMU",
            color = :orange,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in result_validation.values.saveval),
            collect(v[2] for v in result_validation.values.saveval),
            collect(v[3] for v in result_validation.values.saveval),
            label = "Neural FMU",
            color = :blue,
        )
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ e6dc8aab-82c1-4dc9-a1c8-4fe9c137a146
md"""
#### The letter `e`
"""

# ╔═╡ dfee214e-bd13-4d4f-af8e-20e0c4e0de9b
let
    if LIVE_RESULTS
        fig = plot(;
            dpi = 300,
            size = (25 * 10, 30 * 10),
            xlims = (0.265, 0.29),
            ylims = (-0.025, 0.005),
            legend = :topleft,
        )
        plotPaths!(
            fig,
            data_validation.tcp_px,
            data_validation.tcp_py,
            data_validation.tcp_norm_f,
            label = "Data",
            color = :black,
            style = :dash,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in fmu_validation.values.saveval),
            collect(v[2] for v in fmu_validation.values.saveval),
            collect(v[3] for v in fmu_validation.values.saveval),
            label = "FMU",
            color = :orange,
        )
        plotPaths!(
            fig,
            collect(v[1] for v in result_validation.values.saveval),
            collect(v[2] for v in result_validation.values.saveval),
            collect(v[3] for v in result_validation.values.saveval),
            label = "Neural FMU",
            color = :blue,
        )
    else
        LIVE_RESULTS_MESSAGE
    end
end

# ╔═╡ 88884204-79e4-4412-b861-ebeb5f6f7396
md""" 
# Conclusion
Hopefully you got a good first insight in the topic hybrid modeling using FMI and collected your first sense of achievement. Did you find a nice optimum? In case you don't, some rough hyper parameters are given below.

## Hint
If your results are not *that* promising, here is a set of hyperparameters to check. It is *not* a optimal set of parameters, but a *good* set, so feel free to explore the *best*!

| Parameter | Value |
| ----- | ----- |
| eta | 1e-3 |
| layer count | 3 |
| layer width | 32 |
| initial gate opening | 0.2 |
| batch element length | 0.05s |
| training steps | $\geq$ 10 000 |
| additional variables | Joint 1 Angle $br Joint 2 Angle $br TCP velocity x $br TCP velocity y $br TCP nominal force |

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
# ╠═10cb63ad-03d7-47e9-bc33-16c7786b9f6a
# ╟─1e0fa041-a592-42fb-bafd-c7272e346e46
# ╟─6fc16c34-c0c8-48ce-87b3-011a9a0f4e7c
# ╟─8a82d8c7-b781-4600-8780-0a0a003b676c
# ╟─a02f77d1-00d2-46a3-91ba-8a7f5b4bbdc9
# ╠═a1ee798d-c57b-4cc3-9e19-fb607f3e1e43
# ╟─02f0add7-9c4e-4358-8b5e-6863bae3ee75
# ╠═72604eef-5951-4934-844d-d2eb7eb0292c
# ╠═21104cd1-9fe8-45db-9c21-b733258ff155
# ╠═9d9e5139-d27e-48c8-a62e-33b2ae5b0086
# ╟─85308992-04c4-4d20-a840-6220cab54680
# ╠═eaae989a-c9d2-48ca-9ef8-fd0dbff7bcca
# ╠═98c608d9-c60e-4eb6-b611-69d2ae7054c9
# ╟─3e2579c2-39ce-4249-ad75-228f82e616da
# ╠═ddc9ce37-5f93-4851-a74f-8739b38ab092
# ╟─93fab704-a8dd-47ec-ac88-13f32be99460
# ╠═de7a4639-e3b8-4439-924d-7d801b4b3eeb
# ╟─5cb505f7-01bd-4824-8876-3e0f5a922fb7
# ╠═45c4b9dd-0b04-43ae-a715-cd120c571424
# ╠═33d648d3-e66e-488f-a18d-e538ebe9c000
# ╟─1e9541b8-5394-418d-8c27-2831951c538d
# ╠═e6e91a22-7724-46a3-88c1-315c40660290
# ╟─44500f0a-1b89-44af-b135-39ce0fec5810
# ╠═33223393-bfb9-4e9a-8ea6-a3ab6e2f22aa
# ╟─74d23661-751b-4371-bf6b-986149124e81
# ╠═c88b0627-2e04-40ab-baa2-b4c1edfda0c3
# ╟─915e4601-12cc-4b7e-b2fe-574e116f3a92
# ╟─f8e40baa-c1c5-424a-9780-718a42fd2b67
# ╠═74289e0b-1292-41eb-b13b-a4a5763c72b0
# ╟─f111e772-a340-4217-9b63-e7715f773b2c
# ╟─92ad1a99-4ad9-4b69-b6f3-84aab49db54f
# ╟─909de9f1-2aca-4bf0-ba60-d3418964ba4a
# ╟─d8ca5f66-4f55-48ab-a6c9-a0be662811d9
# ╠═41b1c7cb-5e3f-4074-a681-36dd2ef94454
# ╠═8f45871f-f72a-423f-8101-9ce93e5a885b
# ╠═57c039f7-5b24-4d63-b864-d5f808110b91
# ╟─4510022b-ad28-4fc2-836b-e4baf3c14d26
# ╠═9589416a-f9b3-4b17-a381-a4f660a5ee4c
# ╟─326ae469-43ab-4bd7-8dc4-64575f4a4d3e
# ╠═8f8f91cc-9a92-4182-8f18-098ae3e2c553
# ╟─8d93a1ed-28a9-4a77-9ac2-5564be3729a5
# ╠═4a8de267-1bf4-42c2-8dfe-5bfa21d74b7e
# ╟─6a8b98c9-e51a-4f1c-a3ea-cc452b9616b7
# ╟─dbde2da3-e3dc-4b78-8f69-554018533d35
# ╠═d42d0beb-802b-4d30-b5b8-683d76af7c10
# ╟─e50d7cc2-7155-42cf-9fef-93afeee6ffa4
# ╟─3756dd37-03e0-41e9-913e-4b4f183d8b81
# ╠═2f83bc62-5a54-472a-87a2-4ddcefd902b6
# ╟─c228eb10-d694-46aa-b952-01d824879287
# ╟─16ffc610-3c21-40f7-afca-e9da806ea626
# ╠═052f2f19-767b-4ede-b268-fce0aee133ad
# ╟─746fbf6f-ed7c-43b8-8a6f-0377cd3cf85e
# ╟─08e1ff54-d115-4da9-8ea7-5e89289723b3
# ╟─70c6b605-54fa-40a3-8bce-a88daf6a2022
# ╠═634f923a-5e09-42c8-bac0-bf165ab3d12a
# ╟─f59b5c84-2eae-4e3f-aaec-116c090d454d
# ╠═0c9493c4-322e-41a0-9ec7-2e2c54ae1373
# ╟─325c3032-4c78-4408-b86e-d9aa4cfc3187
# ╠═25e55d1c-388f-469d-99e6-2683c0508693
# ╟─74c519c9-0eef-4798-acff-b11044bb4bf1
# ╟─786c4652-583d-43e9-a101-e28c0b6f64e4
# ╟─5d688c3d-b5e3-4a3a-9d91-0896cc001000
# ╠═2e08df84-a468-4e99-a277-e2813dfeae5c
# ╟─68719de3-e11e-4909-99a3-5e05734cc8b1
# ╟─b42bf3d8-e70c-485c-89b3-158eb25d8b25
# ╟─c446ed22-3b23-487d-801e-c23742f81047
# ╠═fc3d7989-ac10-4a82-8777-eeecd354a7d0
# ╟─0a7955e7-7c1a-4396-9613-f8583195c0a8
# ╟─4912d9c9-d68d-4afd-9961-5d8315884f75
# ╟─19942162-cd4e-487c-8073-ea6b262d299d
# ╟─73575386-673b-40cc-b3cb-0b8b4f66a604
# ╟─24861a50-2319-4c63-a800-a0a03279efe2
# ╟─93735dca-c9f3-4f1a-b1bd-dfe312a0644a
# ╟─13ede3cd-99b1-4e65-8a18-9043db544728
# ╟─f7c119dd-c123-4c43-812e-d0625817d77e
# ╟─f4e66f76-76ff-4e21-b4b5-c1ecfd846329
# ╟─b163115b-393d-4589-842d-03859f05be9a
# ╟─ac0afa6c-b6ec-4577-aeb6-10d1ec63fa41
# ╟─5e9cb956-d5ea-4462-a649-b133a77929b0
# ╟─9dc93971-85b6-463b-bd17-43068d57de94
# ╟─476a1ed7-c865-4878-a948-da73d3c81070
# ╟─0b6b4f6d-be09-42f3-bc2c-5f17a8a9ab0e
# ╟─a1aca180-d561-42a3-8d12-88f5a3721aae
# ╟─3bc2b859-d7b1-4b79-88df-8fb517a6929d
# ╟─a501d998-6fd6-496f-9718-3340c42b08a6
# ╟─83a2122d-56da-4a80-8c10-615a8f76c4c1
# ╟─e342be7e-0806-4f72-9e32-6d74ed3ed3f2
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
# ╟─b864631b-a9f3-40d4-a6a8-0b57a37a476d
# ╟─0fb90681-5d04-471a-a7a8-4d0f3ded7bcf
# ╟─95e14ea5-d82d-4044-8c68-090d74d95a61
# ╟─2fa1821b-aaec-4de4-bfb4-89560790dc39
# ╟─cbae6aa4-1338-428c-86aa-61d3304e33ed
# ╟─9b52a65a-f20c-4387-aaca-5292a92fb639
# ╟─8c56acd6-94d3-4cbc-bc29-d249740268a0
# ╟─845a95c4-9a35-44ae-854c-57432200da1a
# ╟─5a399a9b-32d9-4f93-a41f-8f16a4b102dc
# ╟─fd1cebf1-5ccc-4bc5-99d4-1eaa30e9762e
# ╟─1cd976fb-db40-4ebe-b40d-b996e16fc213
# ╟─93771b35-4edd-49e3-bed1-a3ccdb7975e6
# ╟─e79badcd-0396-4a44-9318-8c6b0a94c5c8
# ╟─2a5157c5-f5a2-4330-b2a3-0c1ec0b7adff
# ╟─4454c8d2-68ed-44b4-adfa-432297cdc957
# ╟─d240c95c-5aba-4b47-ab8d-2f9c0eb854cd
# ╟─06937575-9ab1-41cd-960c-7eef3e8cae7f
# ╟─356b6029-de66-418f-8273-6db6464f9fbf
# ╟─5805a216-2536-44ac-a702-d92e86d435a4
# ╟─68d57a23-68c3-418c-9c6f-32bdf8cafceb
# ╟─53e971d8-bf43-41cc-ac2b-20dceaa78667
# ╟─e8b8c63b-2ca4-4e6a-a801-852d6149283e
# ╟─c0ac7902-0716-4f18-9447-d18ce9081ba5
# ╟─84215a73-1ab0-416d-a9db-6b29cd4f5d2a
# ╟─f9d35cfd-4ae5-4dcd-94d9-02aefc99bdfb
# ╟─bc09bd09-2874-431a-bbbb-3d53c632be39
# ╠═f741b213-a20d-423a-a382-75cae1123f2c
# ╟─f02b9118-3fb5-4846-8c08-7e9bbca9d208
# ╠═91473bef-bc23-43ed-9989-34e62166d455
# ╟─404ca10f-d944-4a9f-addb-05efebb4f159
# ╟─d347d51b-743f-4fec-bed7-6cca2b17bacb
# ╟─d60d2561-51a4-4f8a-9819-898d70596e0c
# ╟─c97f2dea-cb18-409d-9ae8-1d03647a6bb3
# ╟─366abd1a-bcb5-480d-b1fb-7c76930dc8fc
# ╟─7e2ffd6f-19b0-435d-8e3c-df24a591bc55
# ╠═caa5e04a-2375-4c56-8072-52c140adcbbb
# ╟─69657be6-6315-4655-81e2-8edef7f21e49
# ╟─23ad65c8-5723-4858-9abe-750c3b65c28a
# ╟─abc57328-4de8-42d8-9e79-dd4020769dd9
# ╟─e8bae97d-9f90-47d2-9263-dc8fc065c3d0
# ╟─2dce68a7-27ec-4ffc-afba-87af4f1cb630
# ╟─c3f5704b-8e98-4c46-be7a-18ab4f139458
# ╟─1a608bc8-7264-4dd3-a4e7-0e39128a8375
# ╟─ff106912-d18c-487f-bcdd-7b7af2112cab
# ╟─51eeb67f-a984-486a-ab8a-a2541966fa72
# ╟─27458e32-5891-4afc-af8e-7afdf7e81cc6
# ╟─737e2c50-0858-4205-bef3-f541e33b85c3
# ╟─5dd491a4-a8cd-4baf-96f7-7a0b850bb26c
# ╟─4f27b6c0-21da-4e26-aaad-ff453c8af3da
# ╟─1195a30c-3b48-4bd2-8a3a-f4f74f3cd864
# ╟─b0ce7b92-93e0-4715-8324-3bf4ff42a0b3
# ╟─919419fe-35de-44bb-89e4-8f8688bee962
# ╟─ed25a535-ca2f-4cd2-b0af-188e9699f1c3
# ╟─2918daf2-6499-4019-a04b-8c3419ee1ab7
# ╟─d798a5d0-3017-4eab-9cdf-ee85d63dfc49
# ╟─048e39c3-a3d9-4e6b-b050-1fd5a919e4ae
# ╟─b489f97d-ee90-48c0-af06-93b66a1f6d2e
# ╟─4dad3e55-5bfd-4315-bb5a-2680e5cbd11c
# ╟─ea0ede8d-7c2c-4e72-9c96-3260dc8d817d
# ╟─35f52dbc-0c0b-495e-8fd4-6edbc6fa811e
# ╟─51aed933-2067-4ea8-9c2f-9d070692ecfc
# ╟─8d9dc86e-f38b-41b1-80c6-b2ab6f488a3a
# ╟─74ef5a39-1dd7-404a-8baf-caa1021d3054
# ╟─347d209b-9d41-48b0-bee6-0d159caacfa9
# ╟─05281c4f-dba8-4070-bce3-dc2f1319902e
# ╟─590d7f24-c6b6-4524-b3db-0c93d9963b74
# ╟─67cfe7c5-8e62-4bf0-996b-19597d5ad5ef
# ╟─e6dc8aab-82c1-4dc9-a1c8-4fe9c137a146
# ╟─dfee214e-bd13-4d4f-af8e-20e0c4e0de9b
# ╟─88884204-79e4-4412-b861-ebeb5f6f7396
