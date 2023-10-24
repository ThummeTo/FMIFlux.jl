# ME-NeuralFMU from the Modelica Conference 2021
Tutorial by Johannes Stoljar, Tobias Thummerer

*Last edit: 29.03.2023*

## License


```julia
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons, Johannes Stoljar
# Licensed under the MIT license. 
# See LICENSE (https://github.com/thummeto/FMIFlux.jl/blob/main/LICENSE) file in the project root for details.
```

## Motivation
The Julia Package *FMIFlux.jl* is motivated by the application of hybrid modeling. This package enables the user to integrate his simulation model between neural networks (NeuralFMU). For this, the simulation model must be exported as FMU (functional mock-up unit), which corresponds to a widely used standard. The big advantage of hybrid modeling with artificial neural networks is, that the effects that are difficult to model (because they might be unknown) can be easily learned by the neural networks. For this purpose, the NeuralFMU is trained with measurement data containing the not modeled physical effect. The final product is a simulation model including the originally not modeled effects. Another big advantage of the NeuralFMU is that it works with little data, because the FMU already contains the characteristic functionality of the simulation and only the missing effects are added.

NeuralFMUs do not need to be as easy as in this example. Basically a NeuralFMU can combine different ANN topologies that manipulate any FMU-input (system state, system inputs, time) and any FMU-output (system state derivative, system outputs, other system variables). However, for this example a NeuralFMU topology as shown in the following picture is used.

![NeuralFMU.svg](https://github.com/thummeto/FMIFlux.jl/blob/main/docs/src/examples/img/NeuralFMU.svg?raw=true)

*NeuralFMU (ME) from* [[1]](#Source).

## Introduction to the example
In this example, simplified modeling of a one-dimensional spring pendulum (without friction) is compared to a model of the same system that includes a nonlinear friction model. The FMU with the simplified model will be named *simpleFMU* in the following and the model with the friction will be named *realFMU*. At the beginning, the actual state of both simulations is shown, whereby clear deviations can be seen in the graphs. In addition, the initial states are changed for both models and these graphs are also contrasted, and the differences can again be clearly seen. The *realFMU* serves as a reference graph. The *simpleFMU* is then integrated into a NeuralFMU architecture and a training of the entire network is performed. After the training the final state is compared again to the *realFMU*. It can be clearly seen that by using the NeuralFMU, learning of the friction process has taken place.  


## Target group
The example is primarily intended for users who work in the field of first principle and/or hybrid modeling and are further interested in hybrid model building. The example wants to show how simple it is to combine FMUs with machine learning and to illustrate the advantages of this approach.


## Other formats
Besides, this [Jupyter Notebook](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021.ipynb) there is also a [Julia file](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021.jl) with the same name, which contains only the code cells. For the documentation there is a [Markdown file](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021.md) corresponding to the notebook.  


## Getting started

### Installation prerequisites
|     | Description                       | Command                   |   
|:----|:----------------------------------|:--------------------------|
| 1.  | Enter Package Manager via         | ]                         |
| 2.  | Install FMI via                   | add FMI                   | 
| 3.  | Install FMIFlux via               | add FMIFlux               | 
| 4.  | Install FMIZoo via                | add FMIZoo                | 
| 5.  | Install DifferentialEquations via | add DifferentialEquations |  
| 6.  | Install Plots via                 | add Plots                 | 
| 7.  | Install Random via                | add Random                | 

## Code section

To run the example, the previously installed packages must be included. 


```julia
# imports
using FMI
using FMIFlux
using FMIFlux.Flux
using FMIZoo
using DifferentialEquations: Tsit5
import Plots

# set seed
import Random
Random.seed!(1234);
```

After importing the packages, the path to the *Functional Mock-up Units* (FMUs) is set. The exported FMU is a model meeting the *Functional Mock-up Interface* (FMI) Standard. The FMI is a free standard ([fmi-standard.org](http://fmi-standard.org/)) that defines a container and an interface to exchange dynamic models using a combination of XML files, binaries and C code zipped into a single file. 

The object-orientated structure of the *SpringPendulum1D* (*simpleFMU*) can be seen in the following graphic and corresponds to a simple modeling.

![svg](https://github.com/thummeto/FMIFlux.jl/blob/main/docs/src/examples/img/SpringPendulum1D.svg?raw=true)

In contrast, the model *SpringFrictionPendulum1D* (*realFMU*) is somewhat more accurate, because it includes a friction component. 

![svg](https://github.com/thummeto/FMIFlux.jl/blob/main/docs/src/examples/img/SpringFrictionPendulum1D.svg?raw=true)

Next, the start time and end time of the simulation are set. Finally, a step size is specified to store the results of the simulation at these time steps.


```julia
tStart = 0.0
tStep = 0.01
tStop = 4.0
tSave = collect(tStart:tStep:tStop)
```




    401-element Vector{Float64}:
     0.0
     0.01
     0.02
     0.03
     0.04
     0.05
     0.06
     0.07
     0.08
     0.09
     0.1
     0.11
     0.12
     ⋮
     3.89
     3.9
     3.91
     3.92
     3.93
     3.94
     3.95
     3.96
     3.97
     3.98
     3.99
     4.0



### RealFMU

In the next lines of code the FMU of the *realFMU* model from *FMIZoo.jl* is loaded and the information about the FMU is shown.  


```julia
realFMU = fmiLoad("SpringFrictionPendulum1D", "Dymola", "2022x")
fmiInfo(realFMU)
```

    #################### Begin information for FMU ####################
    	Model name:			SpringFrictionPendulum1D
    	FMI-Version:			2.0
    	GUID:				{2e178ad3-5e9b-48ec-a7b2-baa5669efc0c}
    	Generation tool:		Dymola Version 2022x (64-bit), 2021-10-08
    	Generation time:		2022-05-19T06:54:12Z
    	Var. naming conv.:		structured
    	Event indicators:		24
    	Inputs:				0
    	Outputs:			0
    	States:				2
    		33554432 ["mass.s"]
    		33554433 ["mass.v", "mass.v_relfric"]
    	Supports Co-Simulation:		true
    		Model identifier:	SpringFrictionPendulum1D
    		Get/Set State:		true
    		Serialize State:	true
    		Dir. Derivatives:	true
    		Var. com. steps:	true
    		Input interpol.:	true
    		Max order out. der.:	1
    	Supports Model-Exchange:	true
    		Model identifier:	SpringFrictionPendulum1D
    		Get/Set State:		true
    		Serialize State:	true
    		Dir. Derivatives:	true
    ##################### End information for FMU #####################
    

In the following two subsections, the *realFMU* is simulated twice with different initial states to show what effect the choice of initial states has.

#### Default initial states

In the next steps the parameters are defined. The first parameter is the initial position of the mass, which is initialized with $0.5m$, the second parameter is the initial velocity, which is initialized with $0\frac{m}{s}$. In the function `fmiSimulate()` the *realFMU* is simulated, still specifying the start and end time, the parameters and which variables are recorded. After the simulation is finished the result of the *realFMU* can be plotted. This plot also serves as a reference for the other model (*simpleFMU*). The extracted data will still be needed later on.


```julia
initStates = ["s0", "v0"]
x₀ = [0.5, 0.0]
params = Dict(zip(initStates, x₀))
vrs = ["mass.s", "mass.v", "mass.a", "mass.f"]

realSimData = fmiSimulate(realFMU, (tStart, tStop); parameters=params, recordValues=vrs, saveat=tSave)
posReal = fmi2GetSolutionValue(realSimData, "mass.s")
velReal = fmi2GetSolutionValue(realSimData, "mass.v")
fmiPlot(realSimData)
```

![svg](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021_files/modelica_conference_2021_12_0.svg?raw=true)  

#### Define functions

The structure of the previous code section is used more often in the further sections, so for clarity the previously explained code section for setting the paramters and simulating are combined into one function `simulate()`.


```julia
function simulate(FMU, initStates, x₀, variables, tStart, tStop, tSave)
    params = Dict(zip(initStates, x₀))
    return fmiSimulate(FMU, (tStart, tStop); parameters=params, recordValues=variables, saveat=tSave)
end
```




    simulate (generic function with 1 method)



Also, a function to extract the position and velocity from the simulation data is created.


```julia
function extractPosVel(simData)
    if simData.states === nothing
        posData = fmi2GetSolutionValue(simData, "mass.s")
        velData = fmi2GetSolutionValue(simData, "mass.v")
    else
        posData = fmi2GetSolutionState(simData, 1; isIndex=true)
        velData = fmi2GetSolutionState(simData, 2; isIndex=true)
    end

    return posData, velData
end
```




    extractPosVel (generic function with 1 method)



#### Modified initial states

In contrast to the previous section, other initial states are selected. The position of the mass is initialized with $1.0m$ and the velocity is initialized with $-1.5\frac{m}{s}$. With the modified initial states the *realFMU* is simulated and a graph is generated.


```julia
xMod₀ = [1.0, -1.5]
realSimDataMod = simulate(realFMU, initStates, xMod₀, vrs, tStart, tStop, tSave)
fmiPlot(realSimDataMod)
```

![svg](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021_files/modelica_conference_2021_18_0.svg?raw=true)

After the plots are created, the FMU is unloaded.


```julia
fmiUnload(realFMU)
```

### SimpleFMU

The following lines load the *simpleFMU* from *FMIZoo.jl*. 


```julia
simpleFMU = fmiLoad("SpringPendulum1D", "Dymola", "2022x")
fmiInfo(simpleFMU)
```

    #################### Begin information for FMU ####################
    	Model name:			SpringPendulum1D
    	FMI-Version:			2.0
    	GUID:				{fc15d8c4-758b-48e6-b00e-5bf47b8b14e5}
    	Generation tool:		Dymola Version 2022x (64-bit), 2021-10-08
    	Generation time:		2022-05-19T06:54:23Z
    	Var. naming conv.:		structured
    	Event indicators:		0
    	Inputs:				0
    	Outputs:			0
    	States:				2
    		33554432 ["mass.s"]
    		33554433 ["mass.v"]
    	Supports Co-Simulation:		true
    		Model identifier:	SpringPendulum1D
    		Get/Set State:		true
    		Serialize State:	true
    		Dir. Derivatives:	true
    		Var. com. steps:	true
    		Input interpol.:	true
    		Max order out. der.:	1
    	Supports Model-Exchange:	true
    		Model identifier:	SpringPendulum1D
    		Get/Set State:		true
    		Serialize State:	true
    		Dir. Derivatives:	true
    ##################### End information for FMU #####################
    

The differences between both systems can be clearly seen from the plots in the subchapters. In the plot for the *realFMU* it can be seen that the oscillation continues to decrease due to the effect of the friction. If you simulate long enough, the oscillation would come to a standstill in a certain time. The oscillation in the *simpleFMU* behaves differently, since the friction was not taken into account here. The oscillation in this model would continue to infinity with the same oscillation amplitude. From this observation the desire of an improvement of this model arises.     


In the following two subsections, the *simpleFMU* is simulated twice with different initial states to show what effect the choice of initial states has.

#### Default initial states

Similar to the simulation of the *realFMU*, the *simpleFMU* is also simulated with the default values for the position and velocity of the mass and then plotted. There is one difference, however, as another state representing a fixed displacement is set. In addition, the last variable is also removed from the variables to be plotted.


```julia
initStates = ["mass_s0", "mass_v0", "fixed.s0"]
displacement = 0.1
xSimple₀ = vcat(x₀, displacement)
vrs = vrs[1:end-1]

simpleSimData = simulate(simpleFMU, initStates, xSimple₀, vrs, tStart, tStop, tSave)
fmiPlot(simpleSimData)
```

![svg](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021_files/modelica_conference_2021_25_0.svg?raw=true)

#### Modified initial states

The same values for the initial states are used for this simulation as for the simulation from the *realFMU* with the modified initial states.


```julia
xSimpleMod₀ = vcat(xMod₀, displacement)

simpleSimDataMod = simulate(simpleFMU, initStates, xSimpleMod₀, vrs, tStart, tStop, tSave)
fmiPlot(simpleSimDataMod)
```

![svg](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021_files/modelica_conference_2021_27_0.svg?raw=true)

## NeuralFMU

#### Loss function

In order to train our model, a loss function must be implemented. The solver of the NeuralFMU can calculate the gradient of the loss function. The gradient descent is needed to adjust the weights in the neural network so that the sum of the error is reduced and the model becomes more accurate.

The error function in this implementation consists of the mean of the mean squared errors. The first part of the addition is the deviation of the position and the second part is the deviation of the velocity. The mean squared error (mse) for the position consists from the real position of the *realFMU* simulation (posReal) and the position data of the network (posNet). The mean squared error for the velocity consists of the real velocity of the *realFMU* simulation (velReal) and the velocity data of the network (velNet).
$$ e_{loss} = \frac{1}{2} \Bigl[ \frac{1}{n} \sum\limits_{i=0}^n (posReal[i] - posNet[i])^2 + \frac{1}{n} \sum\limits_{i=0}^n (velReal[i] - velNet[i])^2 \Bigr]$$


```julia
# loss function for training
function lossSum(p)
    global x₀
    solution = neuralFMU(x₀; p=p)

    posNet, velNet = extractPosVel(solution)

    (FMIFlux.Losses.mse(posReal, posNet) + FMIFlux.Losses.mse(velReal, velNet)) / 2.0
end
```




    lossSum (generic function with 1 method)



#### Callback

To output the loss in certain time intervals, a callback is implemented as a function in the following. Here a counter is incremented, every fiftieth pass the loss function is called and the average error is printed out. Also, the parameters for the velocity in the first layer are kept to a fixed value.


```julia
# callback function for training
global counter = 0
function callb(p)
    global counter
    counter += 1

    # freeze first layer parameters (2,4,6) for velocity -> (static) direct feed trough for velocity
    # parameters for position (1,3,5) are learned
    p[1][2] = 0.0
    p[1][4] = 1.0
    p[1][6] = 0.0

    if counter % 50 == 1
        avgLoss = lossSum(p[1])
        @info "  Loss [$counter]: $(round(avgLoss, digits=5))
        Avg displacement in data: $(round(sqrt(avgLoss), digits=5))
        Weight/Scale: $(paramsNet[1][1])   Bias/Offset: $(paramsNet[1][5])"
    end
end
```




    callb (generic function with 1 method)



#### Functions for plotting

In this section some important functions for plotting are defined. The function `generate_figure()` creates a new figure object and sets some attributes.


```julia
function generate_figure(title, xLabel, yLabel, xlim=:auto)
    Plots.plot(
        title=title, xlabel=xLabel, ylabel=yLabel, linewidth=2,
        xtickfontsize=12, ytickfontsize=12, xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12, legend=:topright, xlim=xlim)
end
```




    generate_figure (generic function with 2 methods)



In the following function, the data of the *realFMU*, *simpleFMU* and *neuralFMU* are summarized and displayed in a graph.


```julia
function plot_results(title, xLabel, yLabel, interval, realData, simpleData, neuralData)
    linestyles = [:dot, :solid]
    
    fig = generate_figure(title, xLabel, yLabel)
    Plots.plot!(fig, interval, simpleData, label="SimpleFMU", linewidth=2)
    Plots.plot!(fig, interval, realData, label="Reference", linewidth=2)
    for i in 1:length(neuralData)
        Plots.plot!(fig, neuralData[i][1], neuralData[i][2], label="NeuralFMU ($(i*2500))", 
                    linewidth=2, linestyle=linestyles[i], linecolor=:green)
    end
    Plots.display(fig)
end
```




    plot_results (generic function with 1 method)



This is the superordinate function, which at the beginning extracts the position and velocity from the simulation data (`realSimData`, `realSimDataMod`, `simpleSimData`,..., `solutionAfterMod`). Four graphs are then generated, each comparing the corresponding data from the *realFMU*, *simpleFMU*, and *neuralFMU*. The comparison is made with the simulation data from the simulation with the default and modified initial states. According to the data, the designation of the title and the naming of the axes is adapted.


```julia
function plot_all_results(realSimData, realSimDataMod, simpleSimData, 
        simpleSimDataMod, solutionAfter, solutionAfterMod)    
    # collect all data
    posReal, velReal = extractPosVel(realSimData)
    posRealMod, velRealMod = extractPosVel(realSimDataMod)
    posSimple, velSimple = extractPosVel(simpleSimData)
    posSimpleMod, velSimpleMod = extractPosVel(simpleSimDataMod)
    
    run = length(solutionAfter)
    
    posNeural, velNeural = [], []
    posNeuralMod, velNeuralMod = [], []
    for i in 1:run
        dataNeural = extractPosVel(solutionAfter[i])
        time = fmi2GetSolutionTime(solutionAfter[i])

        push!(posNeural, (time, dataNeural[1]))
        push!(velNeural, (time, dataNeural[2]))
        
        dataNeuralMod = extractPosVel(solutionAfterMod[i])
        time = fmi2GetSolutionTime(solutionAfterMod[i])
        push!(posNeuralMod, (time, dataNeuralMod[1]))
        push!(velNeuralMod, (time, dataNeuralMod[2]))
    end
         
    # plot results s (default initial states)
    xLabel="t [s]"
    yLabel="mass position [m]"
    title = "Default: Mass position after Run: $(run)"
    plot_results(title, xLabel, yLabel, tSave, posReal, posSimple, posNeural)

    # plot results s (modified initial states)
    title = "Modified: Mass position after Run: $(run)"
    plot_results(title, xLabel, yLabel, tSave, posRealMod, posSimpleMod, posNeuralMod)

    # plot results v (default initial states)
    yLabel="mass velocity [m/s]"
    title = "Default: Mass velocity after Run: $(run)"
    plot_results(title, xLabel, yLabel, tSave, velReal, velSimple, velNeural)

    # plot results v (modified initial states)    
    title = "Modified: Mass velocity after Run: $(run)"
    plot_results(title, xLabel, yLabel, tSave, velRealMod, velSimpleMod, velNeuralMod)
end
```




    plot_all_results (generic function with 1 method)



The function `plot_friction_model()` compares the friction model of the *realFMU*, *simpleFMU* and *neuralFMU*. For this, the velocity and force from the simulation data of the *realFMU* is needed. The force data is calculated with the extracted last layer of the *neuralFMU* to the real velocity in line 9 by iterating over the vector `velReal`. In the next rows, the velocity and force data (if available) for each of the three FMUs are combined into a matrix. The first row of the matrix corresponds to the later x-axis and here the velocity is plotted. The second row corresponds to the y-axis and here the force is plotted. This matrix is sorted and plotted by the first entries (velocity) with the function `sortperm()`. The graph with at least three graphs is plotted in line 33. As output this function has the forces of the *neuralFMU*.


```julia
function plot_friction_model(realSimData, netBottom, forces)    
    linestyles = [:dot, :solid]
    
    velReal = fmi2GetSolutionValue(realSimData, "mass.v")
    forceReal = fmi2GetSolutionValue(realSimData, "mass.f")

    push!(forces, zeros(length(velReal)))
    for i in 1:length(velReal)
        forces[end][i] = -netBottom([velReal[i], 0.0])[2]
    end

    run = length(forces) 
    
    fig = generate_figure("Friction model $(run)", "v [m/s]", "friction force [N]", (-1.25, 1.25))

    fricSimple = hcat(velReal, zeros(length(velReal)))
    fricSimple[sortperm(fricSimple[:, 1]), :]
    Plots.plot!(fig, fricSimple[:,1], fricSimple[:,2], label="SimpleFMU", linewidth=2)

    fricReal = hcat(velReal, forceReal)
    fricReal[sortperm(fricReal[:, 1]), :]
    Plots.plot!(fig, fricReal[:,1], fricReal[:,2], label="reference", linewidth=2)

    for i in 1:run
        fricNeural = hcat(velReal, forces[i])
        fricNeural[sortperm(fricNeural[:, 1]), :]
        Plots.plot!(fig, fricNeural[:,1], fricNeural[:,2], label="NeuralFMU ($(i*2500))", 
                    linewidth=2, linestyle=linestyles[i], linecolor=:green)
        @info "Friction model $i mse: $(FMIFlux.Losses.mse(fricNeural[:,2], fricReal[:,2]))"
    end
    flush(stderr)

    Plots.display(fig)
    
    return forces   
end
```




    plot_friction_model (generic function with 1 method)



The following function is used to display the different displacement modells of the *realFMU*, *simpleFMU* and *neuralFMU*. The displacement of the *realFMU* and *simpleFMU* is very trivial and is only a constant. The position data of the *realFMU* is needed to calculate the displacement. The displacement for the *neuralFMU* is calculated using the first extracted layer of the neural network, subtracting the real position and the displacement of the *simpleFMU*. Also in this function, the graphs of the three FMUs are compared in a plot.


```julia
function plot_displacement_model(realSimData, netTop, displacements, tSave, displacement)
    linestyles = [:dot, :solid]
    
    posReal = fmi2GetSolutionValue(realSimData, "mass.s")
    
    push!(displacements, zeros(length(posReal)))
    for i in 1:length(posReal)
        displacements[end][i] = netTop([posReal[i], 0.0])[1] - posReal[i] - displacement
    end

    run = length(displacements)
    fig = generate_figure("Displacement model $(run)", "t [s]", "displacement [m]")
    Plots.plot!(fig, [tSave[1], tSave[end]], [displacement, displacement], label="simpleFMU", linewidth=2)
    Plots.plot!(fig, [tSave[1], tSave[end]], [0.0, 0.0], label="reference", linewidth=2)
    for i in 1:run
        Plots.plot!(fig, tSave, displacements[i], label="NeuralFMU ($(i*2500))", 
                    linewidth=2, linestyle=linestyles[i], linecolor=:green)
    end

    Plots.display(fig)
    
    return displacements
end
```




    plot_displacement_model (generic function with 1 method)



#### Structure of the NeuralFMU

In the following, the topology of the NeuralFMU is constructed. It consists of a dense layer that has exactly as many inputs and outputs as the model has states `numStates` (and therefore state derivatives). It also sets the initial weights and offsets for the first dense layer, as well as the activation function, which consists of the identity. An input layer follows, which then leads into the *simpleFMU* model. The ME-FMU computes the state derivatives for a given system state. Following the *simpleFMU* is a dense layer that has `numStates` states. The output of this layer consists of 8 output nodes and a *identity* activation function. The next layer has 8 input and output nodes with a *tanh* activation function. The last layer is again a dense layer with 8 input nodes and the number of states as outputs. Here, it is important that no *tanh*-activation function follows, because otherwise the pendulums state values would be limited to the interval $[-1;1]$.


```julia
# NeuralFMU setup
numStates = fmiGetNumberOfStates(simpleFMU)

# diagonal matrix 
initW = zeros(numStates, numStates)
for i in 1:numStates
    initW[i,i] = 1
end

net = Chain(Dense(numStates, numStates,  identity),
            x -> simpleFMU(x=x),
            Dense(numStates, 8, identity),
            Dense(8, 8, tanh),
            Dense(8, numStates))
```




    Chain(
      Dense(2 => 2),                        [90m# 6 parameters[39m
      var"#3#4"(),
      Dense(2 => 8),                        [90m# 24 parameters[39m
      Dense(8 => 8, tanh),                  [90m# 72 parameters[39m
      Dense(8 => 2),                        [90m# 18 parameters[39m
    ) [90m                  # Total: 8 arrays, [39m120 parameters, 992 bytes.



#### Definition of the NeuralFMU

The instantiation of the ME-NeuralFMU is done as a one-liner. The FMU (*simpleFMU*), the structure of the network `net`, start `tStart` and end time `tStop`, the numerical solver `Tsit5()` and the time steps `tSave` for saving are specified.


```julia
neuralFMU = ME_NeuralFMU(simpleFMU, net, (tStart, tStop), Tsit5(); saveat=tSave);
```

#### Plot before training

Here the state trajectory of the *simpleFMU* is recorded. Doesn't really look like a pendulum yet, but the system is random initialized by default. In the plots later on, the effect of learning can be seen.


```julia
solutionBefore = neuralFMU(x₀)
fmiPlot(solutionBefore)
```

![svg](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021_files/modelica_conference_2021_47_0.svg?raw=true)

#### Training of the NeuralFMU

For the training of the NeuralFMU the parameters are extracted. All parameters of the first layer are set to the absolute value.


```julia
# train
paramsNet = FMIFlux.params(neuralFMU)

for i in 1:length(paramsNet[1])
    if paramsNet[1][i] < 0.0 
        paramsNet[1][i] = -paramsNet[1][i]
    end
end
```

The well-known Adam optimizer for minimizing the gradient descent is used as further passing parameters. Additionally, the previously defined loss and callback function as well as a one for the number of epochs are passed. Only one epoch is trained so that the NeuralFMU is precompiled.


```julia
optim = Adam()
FMIFlux.train!(lossSum, paramsNet, Iterators.repeated((), 1), optim; cb=()->callb(paramsNet)) 
```

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1]: 0.64142
    [36m[1m│ [22m[39m        Avg displacement in data: 0.80089
    [36m[1m└ [22m[39m        Weight/Scale: 0.5550727972914903   Bias/Offset: 0.0009999999899993953
    

Some vectors for collecting data are initialized and the number of runs, epochs and iterations are set.


```julia
solutionAfter = []
solutionAfterMod = []
forces = []
displacements = []

numRuns = 2
numEpochs= 5
numIterations = 500;
```

#### Training loop

The code section shown here represents the training loop. The loop is structured so that it has `numRuns` runs, where each run has `numEpochs` epochs, and the training is performed at each epoch with `numIterations` iterations. In lines 9 and 10, the data for the *neuralFMU* for the default and modified initial states are appended to the corresponding vectors. The plots for the opposition of position and velocity is done in line 13 by calling the function `plot_all_results`. In the following lines the last layers are extracted from the *neuralFMU* and formed into an independent network `netBottom`. The parameters for the `netBottom` network come from the original architecture and are shared. In line 20, the new network is used to represent the friction model in a graph. An analogous construction of the next part of the training loop, where here the first layer is taken from the *neuralFMU* and converted to its own network `netTop`. This network is used to record the displacement model. The different graphs are generated for each run and can thus be compared. 


```julia
for run in 1:numRuns
    @time for epoch in 1:numEpochs
        @info "Run: $(run)/$(numRuns)  Epoch: $(epoch)/$(numEpochs)"
        FMIFlux.train!(lossSum, paramsNet, Iterators.repeated((), numIterations), optim; cb=()->callb(paramsNet))
    end
    flush(stderr)
    flush(stdout)
    
    push!(solutionAfter, neuralFMU(x₀))
    push!(solutionAfterMod, neuralFMU(xMod₀))

    # generate all plots for the position and velocity
    plot_all_results(realSimData, realSimDataMod, simpleSimData, simpleSimDataMod, solutionAfter, solutionAfterMod)
    
    # friction model extraction
    layersBottom = neuralFMU.model.layers[3:5]
    netBottom = Chain(layersBottom...)
    transferFlatParams!(netBottom, paramsNet, 7)
    
    forces = plot_friction_model(realSimData, netBottom, forces) 
    
    # displacement model extraction
    layersTop = neuralFMU.model.layers[1:1]
    netTop = Chain(layersTop...)
    transferFlatParams!(netTop, paramsNet, 1)

    displacements = plot_displacement_model(realSimData, netTop, displacements, tSave, displacement)
end
```

    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 1/2  Epoch: 1/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [51]: 0.4549
    [36m[1m│ [22m[39m        Avg displacement in data: 0.67446
    [36m[1m└ [22m[39m        Weight/Scale: 0.6028853123602956   Bias/Offset: 0.048283734061567954
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [101]: 0.39139
    [36m[1m│ [22m[39m        Avg displacement in data: 0.62562
    [36m[1m└ [22m[39m        Weight/Scale: 0.6409651053038355   Bias/Offset: 0.08735843926903719
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [151]: 0.35731
    [36m[1m│ [22m[39m        Avg displacement in data: 0.59775
    [36m[1m└ [22m[39m        Weight/Scale: 0.6705316846536342   Bias/Offset: 0.1191972613731546
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [201]: 0.33753
    [36m[1m│ [22m[39m        Avg displacement in data: 0.58098
    [36m[1m└ [22m[39m        Weight/Scale: 0.6940825401675604   Bias/Offset: 0.145251768074772
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [251]: 0.32536
    [36m[1m│ [22m[39m        Avg displacement in data: 0.5704
    [36m[1m└ [22m[39m        Weight/Scale: 0.7129764282011534   Bias/Offset: 0.16638589837092493
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [301]: 0.31728
    [36m[1m│ [22m[39m        Avg displacement in data: 0.56327
    [36m[1m└ [22m[39m        Weight/Scale: 0.7281070598706306   Bias/Offset: 0.18318816025140622
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [351]: 0.31096
    [36m[1m│ [22m[39m        Avg displacement in data: 0.55763
    [36m[1m└ [22m[39m        Weight/Scale: 0.740113463329563   Bias/Offset: 0.19607878331440823
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [401]: 0.30275
    [36m[1m│ [22m[39m        Avg displacement in data: 0.55023
    [36m[1m└ [22m[39m        Weight/Scale: 0.7494888423538925   Bias/Offset: 0.20521930872516153
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [451]: 0.28769
    [36m[1m│ [22m[39m        Avg displacement in data: 0.53637
    [36m[1m└ [22m[39m        Weight/Scale: 0.7570800789498873   Bias/Offset: 0.21053145126638978
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [501]: 0.23876
    [36m[1m│ [22m[39m        Avg displacement in data: 0.48863
    [36m[1m└ [22m[39m        Weight/Scale: 0.763256901547423   Bias/Offset: 0.21045771856529188
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 1/2  Epoch: 2/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [551]: 0.17541
    [36m[1m│ [22m[39m        Avg displacement in data: 0.41882
    [36m[1m└ [22m[39m        Weight/Scale: 0.7715371474201914   Bias/Offset: 0.21449399824109014
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [601]: 0.03236
    [36m[1m│ [22m[39m        Avg displacement in data: 0.17988
    [36m[1m└ [22m[39m        Weight/Scale: 0.7893108518619958   Bias/Offset: 0.2418753555845266
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [651]: 0.02469
    [36m[1m│ [22m[39m        Avg displacement in data: 0.15714
    [36m[1m└ [22m[39m        Weight/Scale: 0.7845046023606831   Bias/Offset: 0.23235054323344057
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [701]: 0.0205
    [36m[1m│ [22m[39m        Avg displacement in data: 0.14316
    [36m[1m└ [22m[39m        Weight/Scale: 0.7814979363491577   Bias/Offset: 0.2274403748917601
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [751]: 0.01846
    [36m[1m│ [22m[39m        Avg displacement in data: 0.13587
    [36m[1m└ [22m[39m        Weight/Scale: 0.7788922375900735   Bias/Offset: 0.22355709983036756
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [801]: 0.01678
    [36m[1m│ [22m[39m        Avg displacement in data: 0.12954
    [36m[1m└ [22m[39m        Weight/Scale: 0.7767663242115828   Bias/Offset: 0.2207269356135955
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [851]: 0.01653
    [36m[1m│ [22m[39m        Avg displacement in data: 0.12859
    [36m[1m└ [22m[39m        Weight/Scale: 0.7749843307515494   Bias/Offset: 0.2187311910596077
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [901]: 0.01587
    [36m[1m│ [22m[39m        Avg displacement in data: 0.12598
    [36m[1m└ [22m[39m        Weight/Scale: 0.7734037181889805   Bias/Offset: 0.21707807552736116
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [951]: 0.01561
    [36m[1m│ [22m[39m        Avg displacement in data: 0.12493
    [36m[1m└ [22m[39m        Weight/Scale: 0.7719216648691173   Bias/Offset: 0.2157322192513981
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1001]: 0.0151
    [36m[1m│ [22m[39m        Avg displacement in data: 0.12287
    [36m[1m└ [22m[39m        Weight/Scale: 0.7702010562133187   Bias/Offset: 0.21407690113205202
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 1/2  Epoch: 3/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1051]: 0.01451
    [36m[1m│ [22m[39m        Avg displacement in data: 0.12045
    [36m[1m└ [22m[39m        Weight/Scale: 0.7688394671452714   Bias/Offset: 0.21301917582999869
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1101]: 0.01575
    [36m[1m│ [22m[39m        Avg displacement in data: 0.1255
    [36m[1m└ [22m[39m        Weight/Scale: 0.7672673931961634   Bias/Offset: 0.21180371308774382
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1151]: 0.01474
    [36m[1m│ [22m[39m        Avg displacement in data: 0.12141
    [36m[1m└ [22m[39m        Weight/Scale: 0.7658226044431705   Bias/Offset: 0.21092009414997015
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1201]: 0.01381
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11753
    [36m[1m└ [22m[39m        Weight/Scale: 0.7642863980008642   Bias/Offset: 0.20982035577730496
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1251]: 0.01371
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11709
    [36m[1m└ [22m[39m        Weight/Scale: 0.7627087867065679   Bias/Offset: 0.20867498883037655
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1301]: 0.01346
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11604
    [36m[1m└ [22m[39m        Weight/Scale: 0.7613505344452618   Bias/Offset: 0.2079053918906765
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1351]: 0.01311
    [36m[1m│ [22m[39m        Avg displacement in data: 0.1145
    [36m[1m└ [22m[39m        Weight/Scale: 0.7596910622150075   Bias/Offset: 0.2067718953463261
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1401]: 0.01278
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11306
    [36m[1m└ [22m[39m        Weight/Scale: 0.758165981943065   Bias/Offset: 0.2057740211333527
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1451]: 0.01266
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11253
    [36m[1m└ [22m[39m        Weight/Scale: 0.7567852695120381   Bias/Offset: 0.20496869578314184
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1501]: 0.01246
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11164
    [36m[1m└ [22m[39m        Weight/Scale: 0.7552065996491217   Bias/Offset: 0.20394780500232124
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 1/2  Epoch: 4/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1551]: 0.01234
    [36m[1m│ [22m[39m        Avg displacement in data: 0.1111
    [36m[1m└ [22m[39m        Weight/Scale: 0.7537227962507519   Bias/Offset: 0.20299776980687864
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1601]: 0.01199
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10952
    [36m[1m└ [22m[39m        Weight/Scale: 0.7522367489840537   Bias/Offset: 0.20203071922747481
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1651]: 0.01162
    [36m[1m│ [22m[39m        Avg displacement in data: 0.1078
    [36m[1m└ [22m[39m        Weight/Scale: 0.7506878386404157   Bias/Offset: 0.20094145786429501
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1701]: 0.01148
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10716
    [36m[1m└ [22m[39m        Weight/Scale: 0.7492934857801102   Bias/Offset: 0.1999869347144274
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1751]: 0.0113
    [36m[1m│ [22m[39m        Avg displacement in data: 0.1063
    [36m[1m└ [22m[39m        Weight/Scale: 0.7478998926577842   Bias/Offset: 0.19906239975398352
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1801]: 0.01111
    [36m[1m│ [22m[39m        Avg displacement in data: 0.1054
    [36m[1m└ [22m[39m        Weight/Scale: 0.7464778719101004   Bias/Offset: 0.1980770525768582
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1851]: 0.01095
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10462
    [36m[1m└ [22m[39m        Weight/Scale: 0.7451455425305433   Bias/Offset: 0.1971590848273303
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1901]: 0.01078
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10383
    [36m[1m└ [22m[39m        Weight/Scale: 0.7438713578573728   Bias/Offset: 0.19628785255503695
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1951]: 0.01098
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10479
    [36m[1m└ [22m[39m        Weight/Scale: 0.7425175124165783   Bias/Offset: 0.19545014923386186
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2001]: 0.01039
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10195
    [36m[1m└ [22m[39m        Weight/Scale: 0.7413631836744611   Bias/Offset: 0.1943805753368975
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 1/2  Epoch: 5/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2051]: 0.00994
    [36m[1m│ [22m[39m        Avg displacement in data: 0.0997
    [36m[1m└ [22m[39m        Weight/Scale: 0.7398841574829385   Bias/Offset: 0.19311827659238118
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2101]: 0.00972
    [36m[1m│ [22m[39m        Avg displacement in data: 0.0986
    [36m[1m└ [22m[39m        Weight/Scale: 0.7389464549605831   Bias/Offset: 0.19237963539961564
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2151]: 0.00963
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09813
    [36m[1m└ [22m[39m        Weight/Scale: 0.7382470389262071   Bias/Offset: 0.19196777879935928
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2201]: 0.00929
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09637
    [36m[1m└ [22m[39m        Weight/Scale: 0.7369374279270299   Bias/Offset: 0.19080399946860455
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2251]: 0.00902
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09497
    [36m[1m└ [22m[39m        Weight/Scale: 0.7360137518688309   Bias/Offset: 0.18988935786719996
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2301]: 0.00876
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09361
    [36m[1m└ [22m[39m        Weight/Scale: 0.7352447333198371   Bias/Offset: 0.18910493732262174
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2351]: 0.00847
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09205
    [36m[1m└ [22m[39m        Weight/Scale: 0.7344242742835179   Bias/Offset: 0.1883314622788676
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2401]: 0.00834
    [36m[1m│ [22m[39m        Avg displacement in data: 0.0913
    [36m[1m└ [22m[39m        Weight/Scale: 0.7336210186424724   Bias/Offset: 0.18782939477657412
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2451]: 0.00803
    [36m[1m│ [22m[39m        Avg displacement in data: 0.0896
    [36m[1m└ [22m[39m        Weight/Scale: 0.7325198837866396   Bias/Offset: 0.18710302833996387
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2501]: 0.00777
    [36m[1m│ [22m[39m        Avg displacement in data: 0.08817
    [36m[1m└ [22m[39m        Weight/Scale: 0.7312404890836444   Bias/Offset: 0.1861703127251153
    

    2408.631332 seconds (12.78 G allocations: 676.655 GiB, 10.69% gc time, 0.28% compilation time)
    
![svg](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021_files/modelica_conference_2021_55_2.svg?raw=true)

![svg](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021_files/modelica_conference_2021_55_3.svg?raw=true)

![svg](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021_files/modelica_conference_2021_55_4.svg?raw=true)

![svg](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021_files/modelica_conference_2021_55_5.svg?raw=true)

    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mFriction model 1 mse: 14.88774263981997

![svg](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021_files/modelica_conference_2021_55_7.svg?raw=true)

![svg](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021_files/modelica_conference_2021_55_8.svg?raw=true)

    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 2/2  Epoch: 1/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2551]: 0.00758
    [36m[1m│ [22m[39m        Avg displacement in data: 0.08704
    [36m[1m└ [22m[39m        Weight/Scale: 0.7303438794480103   Bias/Offset: 0.1856946862923494
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2601]: 0.00709
    [36m[1m│ [22m[39m        Avg displacement in data: 0.08423
    [36m[1m└ [22m[39m        Weight/Scale: 0.7290173198128315   Bias/Offset: 0.18492263168217266
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2651]: 0.00668
    [36m[1m│ [22m[39m        Avg displacement in data: 0.08173
    [36m[1m└ [22m[39m        Weight/Scale: 0.7279708837467755   Bias/Offset: 0.184157848384496
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2701]: 0.00598
    [36m[1m│ [22m[39m        Avg displacement in data: 0.07732
    [36m[1m└ [22m[39m        Weight/Scale: 0.7272986006716042   Bias/Offset: 0.18353867692922435
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2751]: 0.00507
    [36m[1m│ [22m[39m        Avg displacement in data: 0.07123
    [36m[1m└ [22m[39m        Weight/Scale: 0.7275168625544582   Bias/Offset: 0.18329458801206952
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2801]: 0.00448
    [36m[1m│ [22m[39m        Avg displacement in data: 0.06691
    [36m[1m└ [22m[39m        Weight/Scale: 0.7293543567720375   Bias/Offset: 0.18406020756435157
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2851]: 0.00366
    [36m[1m│ [22m[39m        Avg displacement in data: 0.06047
    [36m[1m└ [22m[39m        Weight/Scale: 0.7314497388192254   Bias/Offset: 0.18520834506746972
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2901]: 0.0033
    [36m[1m│ [22m[39m        Avg displacement in data: 0.05748
    [36m[1m└ [22m[39m        Weight/Scale: 0.7335176207351253   Bias/Offset: 0.18690306932839615
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2951]: 0.00295
    [36m[1m│ [22m[39m        Avg displacement in data: 0.05436
    [36m[1m└ [22m[39m        Weight/Scale: 0.7348771164129421   Bias/Offset: 0.18828785164713352
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3001]: 0.00277
    [36m[1m│ [22m[39m        Avg displacement in data: 0.05263
    [36m[1m└ [22m[39m        Weight/Scale: 0.7360374169527821   Bias/Offset: 0.1898477031751862
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 2/2  Epoch: 2/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3051]: 0.00248
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04981
    [36m[1m└ [22m[39m        Weight/Scale: 0.736866128249379   Bias/Offset: 0.1912240818628819
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3101]: 0.00245
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04952
    [36m[1m└ [22m[39m        Weight/Scale: 0.7373855732294173   Bias/Offset: 0.19240528541444005
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3151]: 0.00238
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04878
    [36m[1m└ [22m[39m        Weight/Scale: 0.7378932980674202   Bias/Offset: 0.19366676386495177
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3201]: 0.00227
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04761
    [36m[1m└ [22m[39m        Weight/Scale: 0.7382251175478393   Bias/Offset: 0.19476828949472968
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3251]: 0.00214
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04628
    [36m[1m└ [22m[39m        Weight/Scale: 0.7384893980564785   Bias/Offset: 0.19579811328921606
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3301]: 0.002
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04469
    [36m[1m└ [22m[39m        Weight/Scale: 0.7387185771931547   Bias/Offset: 0.19671211460499438
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3351]: 0.00172
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04143
    [36m[1m└ [22m[39m        Weight/Scale: 0.7389053670518165   Bias/Offset: 0.19768562161724554
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3401]: 0.00181
    [36m[1m│ [22m[39m        Avg displacement in data: 0.0425
    [36m[1m└ [22m[39m        Weight/Scale: 0.7389359668253536   Bias/Offset: 0.19854205640438413
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3451]: 0.00168
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04093
    [36m[1m└ [22m[39m        Weight/Scale: 0.7389763832299235   Bias/Offset: 0.19933020424433798
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3501]: 0.00164
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04047
    [36m[1m└ [22m[39m        Weight/Scale: 0.739133606159328   Bias/Offset: 0.2002524737276869
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 2/2  Epoch: 3/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3551]: 0.0017
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04121
    [36m[1m└ [22m[39m        Weight/Scale: 0.7390849605375236   Bias/Offset: 0.20079114606935808
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3601]: 0.00147
    [36m[1m│ [22m[39m        Avg displacement in data: 0.0384
    [36m[1m└ [22m[39m        Weight/Scale: 0.738717762508065   Bias/Offset: 0.20087666561392792
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3651]: 0.00137
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03707
    [36m[1m└ [22m[39m        Weight/Scale: 0.738422579351483   Bias/Offset: 0.2013353639538088
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3701]: 0.00131
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03618
    [36m[1m└ [22m[39m        Weight/Scale: 0.7382023344717665   Bias/Offset: 0.2018144025618835
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3751]: 0.00133
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03645
    [36m[1m└ [22m[39m        Weight/Scale: 0.7380810438931842   Bias/Offset: 0.2023967059544771
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3801]: 0.0012
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03468
    [36m[1m└ [22m[39m        Weight/Scale: 0.7379229436055246   Bias/Offset: 0.20290101550439926
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3851]: 0.00119
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03447
    [36m[1m└ [22m[39m        Weight/Scale: 0.7378414770298785   Bias/Offset: 0.20353443952846859
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3901]: 0.00114
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03369
    [36m[1m└ [22m[39m        Weight/Scale: 0.7375617453294023   Bias/Offset: 0.20391553159680062
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3951]: 0.00103
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03209
    [36m[1m└ [22m[39m        Weight/Scale: 0.737446244888749   Bias/Offset: 0.20440150307653776
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4001]: 0.00096
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03103
    [36m[1m└ [22m[39m        Weight/Scale: 0.7374493903607047   Bias/Offset: 0.2050175127869096
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 2/2  Epoch: 4/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4051]: 0.00094
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03072
    [36m[1m└ [22m[39m        Weight/Scale: 0.7374186605535206   Bias/Offset: 0.20568745327685167
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4101]: 0.00091
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03016
    [36m[1m└ [22m[39m        Weight/Scale: 0.7372473848007951   Bias/Offset: 0.2062290121465478
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4151]: 0.00087
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02949
    [36m[1m└ [22m[39m        Weight/Scale: 0.7370249736192905   Bias/Offset: 0.20669971773804788
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4201]: 0.00085
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02907
    [36m[1m└ [22m[39m        Weight/Scale: 0.736894244330619   Bias/Offset: 0.20732687291256477
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4251]: 0.0008
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02824
    [36m[1m└ [22m[39m        Weight/Scale: 0.7367357257440418   Bias/Offset: 0.20787120730893408
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4301]: 0.00075
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02746
    [36m[1m└ [22m[39m        Weight/Scale: 0.7365668383366816   Bias/Offset: 0.20839685243021566
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4351]: 0.00072
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02678
    [36m[1m└ [22m[39m        Weight/Scale: 0.7363802850820378   Bias/Offset: 0.20891048885984057
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4401]: 0.00068
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02616
    [36m[1m└ [22m[39m        Weight/Scale: 0.736177689062514   Bias/Offset: 0.20940810376156005
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4451]: 0.00066
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02559
    [36m[1m└ [22m[39m        Weight/Scale: 0.7359671111916609   Bias/Offset: 0.2098926181322704
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4501]: 0.00063
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02508
    [36m[1m└ [22m[39m        Weight/Scale: 0.7357482126785773   Bias/Offset: 0.21036240452912788
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 2/2  Epoch: 5/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4551]: 0.00061
    [36m[1m│ [22m[39m        Avg displacement in data: 0.0248
    [36m[1m└ [22m[39m        Weight/Scale: 0.7355093982927685   Bias/Offset: 0.21081128080699119
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4601]: 0.00059
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02432
    [36m[1m└ [22m[39m        Weight/Scale: 0.7352273138958729   Bias/Offset: 0.21119507003308016
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4651]: 0.00057
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02387
    [36m[1m└ [22m[39m        Weight/Scale: 0.7349675691873173   Bias/Offset: 0.2115914494309266
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4701]: 0.00055
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02345
    [36m[1m└ [22m[39m        Weight/Scale: 0.734711326205178   Bias/Offset: 0.2119878114037143
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4751]: 0.00053
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02306
    [36m[1m└ [22m[39m        Weight/Scale: 0.7344511000104165   Bias/Offset: 0.21237799523812428
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4801]: 0.00051
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02268
    [36m[1m└ [22m[39m        Weight/Scale: 0.7341857977412664   Bias/Offset: 0.2127609918395641
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4851]: 0.0005
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02233
    [36m[1m└ [22m[39m        Weight/Scale: 0.7339151056253107   Bias/Offset: 0.21313685247477435
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4901]: 0.00048
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02199
    [36m[1m└ [22m[39m        Weight/Scale: 0.7336390969886643   Bias/Offset: 0.2135063570250628
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4951]: 0.00047
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02167
    [36m[1m└ [22m[39m        Weight/Scale: 0.7333575153610334   Bias/Offset: 0.21387041951226915
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [5001]: 0.00046
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02145
    [36m[1m└ [22m[39m        Weight/Scale: 0.7330234329581672   Bias/Offset: 0.21420252046915342
    

    2308.455766 seconds (12.39 G allocations: 657.589 GiB, 11.65% gc time)
    
![svg](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021_files/modelica_conference_2021_55_11.svg?raw=true)

![svg](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021_files/modelica_conference_2021_55_12.svg?raw=true)

![svg](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021_files/modelica_conference_2021_55_13.svg?raw=true)

![svg](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021_files/modelica_conference_2021_55_14.svg?raw=true)

    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mFriction model 1 mse: 14.88774263981997
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mFriction model 2 mse: 18.510075764233882

![svg](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021_files/modelica_conference_2021_55_16.svg?raw=true)

![svg](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/modelica_conference_2021_files/modelica_conference_2021_55_17.svg?raw=true)

Finally, the FMU is cleaned-up.


```julia
fmiUnload(simpleFMU)
```

### Summary

Based on the plots, it can be seen that the curves of the *realFMU* and the *neuralFMU* are very close. The *neuralFMU* is able to learn the friction and displacement model.

### Source

[1] Tobias Thummerer, Lars Mikelsons and Josef Kircher. 2021. **NeuralFMU: towards structural integration of FMUs into neural networks.** Martin Sjölund, Lena Buffoni, Adrian Pop and Lennart Ochel (Ed.). Proceedings of 14th Modelica Conference 2021, Linköping, Sweden, September 20-24, 2021. Linköping University Electronic Press, Linköping (Linköping Electronic Conference Proceedings ; 181), 297-306. [DOI: 10.3384/ecp21181297](https://doi.org/10.3384/ecp21181297)

