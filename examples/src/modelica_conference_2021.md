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

    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mPrecompiling FMI [14a09403-18e3-468f-ad8a-74f8dda2d9ac]
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mPrecompiling FMIFlux [fabad875-0d53-4e47-9446-963b74cae21f]
    

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




    
![svg](modelica_conference_2021_files/modelica_conference_2021_12_0.svg)
    



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




    
![svg](modelica_conference_2021_files/modelica_conference_2021_18_0.svg)
    



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




    
![svg](modelica_conference_2021_files/modelica_conference_2021_25_0.svg)
    



#### Modified initial states

The same values for the initial states are used for this simulation as for the simulation from the *realFMU* with the modified initial states.


```julia
xSimpleMod₀ = vcat(xMod₀, displacement)

simpleSimDataMod = simulate(simpleFMU, initStates, xSimpleMod₀, vrs, tStart, tStop, tSave)
fmiPlot(simpleSimDataMod)
```




    
![svg](modelica_conference_2021_files/modelica_conference_2021_27_0.svg)
    



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
      var"#1#2"(),
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




    
![svg](modelica_conference_2021_files/modelica_conference_2021_47_0.svg)
    



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
    [36m[1m└ [22m[39m        Weight/Scale: 0.5550727972914903   Bias/Offset: 0.0009999999899993858
    

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
    [36m[1m└ [22m[39m        Weight/Scale: 0.602885316980542   Bias/Offset: 0.048283738456848496
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [101]: 0.39139
    [36m[1m│ [22m[39m        Avg displacement in data: 0.62562
    [36m[1m└ [22m[39m        Weight/Scale: 0.6409651043273152   Bias/Offset: 0.08735843926521016
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [151]: 0.35731
    [36m[1m│ [22m[39m        Avg displacement in data: 0.59775
    [36m[1m└ [22m[39m        Weight/Scale: 0.6705316811240131   Bias/Offset: 0.1191972586560209
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [201]: 0.33753
    [36m[1m│ [22m[39m        Avg displacement in data: 0.58098
    [36m[1m└ [22m[39m        Weight/Scale: 0.6940825365416404   Bias/Offset: 0.14525176544797544
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [251]: 0.32536
    [36m[1m│ [22m[39m        Avg displacement in data: 0.5704
    [36m[1m└ [22m[39m        Weight/Scale: 0.712976424058577   Bias/Offset: 0.16638589530322162
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [301]: 0.31728
    [36m[1m│ [22m[39m        Avg displacement in data: 0.56327
    [36m[1m└ [22m[39m        Weight/Scale: 0.7281070554547063   Bias/Offset: 0.18318815711239766
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [351]: 0.31096
    [36m[1m│ [22m[39m        Avg displacement in data: 0.55763
    [36m[1m└ [22m[39m        Weight/Scale: 0.7401134587970515   Bias/Offset: 0.1960787802740073
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [401]: 0.30275
    [36m[1m│ [22m[39m        Avg displacement in data: 0.55023
    [36m[1m└ [22m[39m        Weight/Scale: 0.749488838037055   Bias/Offset: 0.2052193059250787
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [451]: 0.28769
    [36m[1m│ [22m[39m        Avg displacement in data: 0.53637
    [36m[1m└ [22m[39m        Weight/Scale: 0.7570800749703885   Bias/Offset: 0.21053144932789805
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [501]: 0.23876
    [36m[1m│ [22m[39m        Avg displacement in data: 0.48863
    [36m[1m└ [22m[39m        Weight/Scale: 0.7632568981431536   Bias/Offset: 0.21045771862932025
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 1/2  Epoch: 2/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [551]: 0.17541
    [36m[1m│ [22m[39m        Avg displacement in data: 0.41882
    [36m[1m└ [22m[39m        Weight/Scale: 0.7715371427920885   Bias/Offset: 0.21449399767486593
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [601]: 0.03236
    [36m[1m│ [22m[39m        Avg displacement in data: 0.17988
    [36m[1m└ [22m[39m        Weight/Scale: 0.7893108434334031   Bias/Offset: 0.24187534657638277
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [651]: 0.02469
    [36m[1m│ [22m[39m        Avg displacement in data: 0.15714
    [36m[1m└ [22m[39m        Weight/Scale: 0.7845046116181404   Bias/Offset: 0.23235054311532496
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [701]: 0.0205
    [36m[1m│ [22m[39m        Avg displacement in data: 0.14316
    [36m[1m└ [22m[39m        Weight/Scale: 0.7814979367914117   Bias/Offset: 0.22744035324323336
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [751]: 0.01846
    [36m[1m│ [22m[39m        Avg displacement in data: 0.13587
    [36m[1m└ [22m[39m        Weight/Scale: 0.7788922351560036   Bias/Offset: 0.22355707643057776
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [801]: 0.01753
    [36m[1m│ [22m[39m        Avg displacement in data: 0.1324
    [36m[1m└ [22m[39m        Weight/Scale: 0.7767614844067194   Bias/Offset: 0.22071784237115313
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [851]: 0.01685
    [36m[1m│ [22m[39m        Avg displacement in data: 0.1298
    [36m[1m└ [22m[39m        Weight/Scale: 0.7749652430778368   Bias/Offset: 0.21879829728398761
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [901]: 0.01593
    [36m[1m│ [22m[39m        Avg displacement in data: 0.1262
    [36m[1m└ [22m[39m        Weight/Scale: 0.773457904717924   Bias/Offset: 0.21726944542613494
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [951]: 0.01569
    [36m[1m│ [22m[39m        Avg displacement in data: 0.12524
    [36m[1m└ [22m[39m        Weight/Scale: 0.7720820921408393   Bias/Offset: 0.21612990312918426
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1001]: 0.01518
    [36m[1m│ [22m[39m        Avg displacement in data: 0.12323
    [36m[1m└ [22m[39m        Weight/Scale: 0.7703716351910177   Bias/Offset: 0.21455299298444633
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 1/2  Epoch: 3/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1051]: 0.01478
    [36m[1m│ [22m[39m        Avg displacement in data: 0.12159
    [36m[1m└ [22m[39m        Weight/Scale: 0.7688856933982932   Bias/Offset: 0.21331990011747196
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1101]: 0.01475
    [36m[1m│ [22m[39m        Avg displacement in data: 0.12143
    [36m[1m└ [22m[39m        Weight/Scale: 0.7673156473731901   Bias/Offset: 0.21213897946641008
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1151]: 0.01427
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11947
    [36m[1m└ [22m[39m        Weight/Scale: 0.7658100555579398   Bias/Offset: 0.2111051096087789
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1201]: 0.01382
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11754
    [36m[1m└ [22m[39m        Weight/Scale: 0.7643855517906682   Bias/Offset: 0.2101963972993835
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1251]: 0.01365
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11683
    [36m[1m└ [22m[39m        Weight/Scale: 0.7627594424020064   Bias/Offset: 0.2090207000876991
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1301]: 0.01336
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11557
    [36m[1m└ [22m[39m        Weight/Scale: 0.7612832893005562   Bias/Offset: 0.20809915014868682
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1351]: 0.01312
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11456
    [36m[1m└ [22m[39m        Weight/Scale: 0.7597710535523635   Bias/Offset: 0.20714257311572343
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1401]: 0.01287
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11347
    [36m[1m└ [22m[39m        Weight/Scale: 0.7581256184109563   Bias/Offset: 0.20603003979342796
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1451]: 0.0126
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11226
    [36m[1m└ [22m[39m        Weight/Scale: 0.7565630420306814   Bias/Offset: 0.2049841386322073
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1501]: 0.01246
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11161
    [36m[1m└ [22m[39m        Weight/Scale: 0.7549658511332776   Bias/Offset: 0.20390377926915973
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 1/2  Epoch: 4/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1551]: 0.01232
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11102
    [36m[1m└ [22m[39m        Weight/Scale: 0.7534637923539111   Bias/Offset: 0.20298187359998485
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1601]: 0.0119
    [36m[1m│ [22m[39m        Avg displacement in data: 0.1091
    [36m[1m└ [22m[39m        Weight/Scale: 0.7518337743863314   Bias/Offset: 0.20182815324697526
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1651]: 0.0116
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10769
    [36m[1m└ [22m[39m        Weight/Scale: 0.7503582808714246   Bias/Offset: 0.2007637217329351
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1701]: 0.01136
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10659
    [36m[1m└ [22m[39m        Weight/Scale: 0.74892200412913   Bias/Offset: 0.19974564916218396
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1751]: 0.0113
    [36m[1m│ [22m[39m        Avg displacement in data: 0.1063
    [36m[1m└ [22m[39m        Weight/Scale: 0.74761458260084   Bias/Offset: 0.1989031575601823
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1801]: 0.01111
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10539
    [36m[1m└ [22m[39m        Weight/Scale: 0.7461684788768355   Bias/Offset: 0.1979132198834549
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1851]: 0.01211
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11003
    [36m[1m└ [22m[39m        Weight/Scale: 0.7446483166160223   Bias/Offset: 0.1968161576733014
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1901]: 0.01091
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10446
    [36m[1m└ [22m[39m        Weight/Scale: 0.7434441375648992   Bias/Offset: 0.19600335272450134
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1951]: 0.01068
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10337
    [36m[1m└ [22m[39m        Weight/Scale: 0.7422256713564275   Bias/Offset: 0.19509681549273405
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2001]: 0.01038
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10187
    [36m[1m└ [22m[39m        Weight/Scale: 0.7412086463753034   Bias/Offset: 0.19449685460124921
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 1/2  Epoch: 5/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2051]: 0.0101
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10049
    [36m[1m└ [22m[39m        Weight/Scale: 0.7399658401479723   Bias/Offset: 0.1935730083556091
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2101]: 0.0098
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09898
    [36m[1m└ [22m[39m        Weight/Scale: 0.7387782583724326   Bias/Offset: 0.19255684736858555
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2151]: 0.00979
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09897
    [36m[1m└ [22m[39m        Weight/Scale: 0.7381285979342257   Bias/Offset: 0.19212567338557288
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2201]: 0.00934
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09664
    [36m[1m└ [22m[39m        Weight/Scale: 0.7367540860940136   Bias/Offset: 0.19095261836584457
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2251]: 0.00909
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09534
    [36m[1m└ [22m[39m        Weight/Scale: 0.735792478265287   Bias/Offset: 0.19003615064047097
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2301]: 0.0089
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09432
    [36m[1m└ [22m[39m        Weight/Scale: 0.7350853661318819   Bias/Offset: 0.1893408146282273
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2351]: 0.00865
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09299
    [36m[1m└ [22m[39m        Weight/Scale: 0.7343391460037718   Bias/Offset: 0.18864684220466435
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2401]: 0.00841
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09169
    [36m[1m└ [22m[39m        Weight/Scale: 0.7335077989234551   Bias/Offset: 0.1879728892157918
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2451]: 0.00832
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09121
    [36m[1m└ [22m[39m        Weight/Scale: 0.7324833256118775   Bias/Offset: 0.18732871993858377
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2501]: 0.00821
    [36m[1m│ [22m[39m        Avg displacement in data: 0.0906
    [36m[1m└ [22m[39m        Weight/Scale: 0.7314249364314245   Bias/Offset: 0.18672447683333782
    

    1342.527815 seconds (10.91 G allocations: 538.852 GiB, 8.03% gc time, 0.15% compilation time)
    


    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_2.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_3.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_4.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_5.svg)
    


    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mFriction model 1 mse: 14.780632206979849
    


    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_7.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_8.svg)
    


    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 2/2  Epoch: 1/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2551]: 0.00779
    [36m[1m│ [22m[39m        Avg displacement in data: 0.08828
    [36m[1m└ [22m[39m        Weight/Scale: 0.730262568461759   Bias/Offset: 0.1859799380679318
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2601]: 0.00752
    [36m[1m│ [22m[39m        Avg displacement in data: 0.08669
    [36m[1m└ [22m[39m        Weight/Scale: 0.7291235395563382   Bias/Offset: 0.18537367060189736
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2651]: 0.00715
    [36m[1m│ [22m[39m        Avg displacement in data: 0.08456
    [36m[1m└ [22m[39m        Weight/Scale: 0.7277387217626177   Bias/Offset: 0.1844470791117029
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2701]: 0.00673
    [36m[1m│ [22m[39m        Avg displacement in data: 0.08202
    [36m[1m└ [22m[39m        Weight/Scale: 0.7268002675886042   Bias/Offset: 0.18395563095261067
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2751]: 0.00607
    [36m[1m│ [22m[39m        Avg displacement in data: 0.07788
    [36m[1m└ [22m[39m        Weight/Scale: 0.7260487554715731   Bias/Offset: 0.18334217262654526
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2801]: 0.00518
    [36m[1m│ [22m[39m        Avg displacement in data: 0.07197
    [36m[1m└ [22m[39m        Weight/Scale: 0.7260096886418587   Bias/Offset: 0.1829716147418429
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2851]: 0.00485
    [36m[1m│ [22m[39m        Avg displacement in data: 0.06962
    [36m[1m└ [22m[39m        Weight/Scale: 0.7274474217413207   Bias/Offset: 0.183488943365658
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2901]: 0.00371
    [36m[1m│ [22m[39m        Avg displacement in data: 0.06091
    [36m[1m└ [22m[39m        Weight/Scale: 0.7296796857092619   Bias/Offset: 0.1846234425679058
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2951]: 0.00336
    [36m[1m│ [22m[39m        Avg displacement in data: 0.05792
    [36m[1m└ [22m[39m        Weight/Scale: 0.7320080941696712   Bias/Offset: 0.18647637065162503
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3001]: 0.00292
    [36m[1m│ [22m[39m        Avg displacement in data: 0.05401
    [36m[1m└ [22m[39m        Weight/Scale: 0.7336219552146461   Bias/Offset: 0.18806700202666357
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 2/2  Epoch: 2/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3051]: 0.00278
    [36m[1m│ [22m[39m        Avg displacement in data: 0.05272
    [36m[1m└ [22m[39m        Weight/Scale: 0.7347321069811726   Bias/Offset: 0.1895195478434213
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3101]: 0.00261
    [36m[1m│ [22m[39m        Avg displacement in data: 0.05105
    [36m[1m└ [22m[39m        Weight/Scale: 0.7356352301204249   Bias/Offset: 0.19098614323886864
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3151]: 0.00245
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04952
    [36m[1m└ [22m[39m        Weight/Scale: 0.7362972672213735   Bias/Offset: 0.19234711385088296
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3201]: 0.00233
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04824
    [36m[1m└ [22m[39m        Weight/Scale: 0.7367508562654395   Bias/Offset: 0.19351228573232834
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3251]: 0.00225
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04747
    [36m[1m└ [22m[39m        Weight/Scale: 0.7371399683595928   Bias/Offset: 0.19469553758838873
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3301]: 0.00215
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04633
    [36m[1m└ [22m[39m        Weight/Scale: 0.7374086323316941   Bias/Offset: 0.19576853938133534
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3351]: 0.00204
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04511
    [36m[1m└ [22m[39m        Weight/Scale: 0.7376309494591057   Bias/Offset: 0.19678581739337486
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3401]: 0.00184
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04291
    [36m[1m└ [22m[39m        Weight/Scale: 0.7377631788076989   Bias/Offset: 0.19765977034125515
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3451]: 0.00181
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04259
    [36m[1m└ [22m[39m        Weight/Scale: 0.7379349886362084   Bias/Offset: 0.19864150907342054
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3501]: 0.0017
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04125
    [36m[1m└ [22m[39m        Weight/Scale: 0.7379664969098976   Bias/Offset: 0.19946091282942022
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 2/2  Epoch: 3/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3551]: 0.00158
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03977
    [36m[1m└ [22m[39m        Weight/Scale: 0.7380561411564669   Bias/Offset: 0.20031282119818805
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3601]: 0.00128
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03575
    [36m[1m└ [22m[39m        Weight/Scale: 0.737980473272092   Bias/Offset: 0.20094201364595074
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3651]: 0.00146
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03818
    [36m[1m└ [22m[39m        Weight/Scale: 0.7380025072203422   Bias/Offset: 0.2016697057498728
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3701]: 0.00134
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03654
    [36m[1m└ [22m[39m        Weight/Scale: 0.7377332953731529   Bias/Offset: 0.20215231108279746
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3751]: 0.00129
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03585
    [36m[1m└ [22m[39m        Weight/Scale: 0.7376751251979092   Bias/Offset: 0.20282831383037178
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3801]: 0.00122
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03496
    [36m[1m└ [22m[39m        Weight/Scale: 0.737537210206008   Bias/Offset: 0.20341503585488435
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3851]: 0.00115
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03394
    [36m[1m└ [22m[39m        Weight/Scale: 0.7374727733582238   Bias/Offset: 0.20407349922449455
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3901]: 0.00121
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03476
    [36m[1m└ [22m[39m        Weight/Scale: 0.7374510805532917   Bias/Offset: 0.20483848909831315
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3951]: 0.00107
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03265
    [36m[1m└ [22m[39m        Weight/Scale: 0.7371871663358004   Bias/Offset: 0.2052272837008678
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4001]: 0.00094
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03067
    [36m[1m└ [22m[39m        Weight/Scale: 0.737070407033106   Bias/Offset: 0.205693660322591
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 2/2  Epoch: 4/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4051]: 0.00089
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02976
    [36m[1m└ [22m[39m        Weight/Scale: 0.7371058744894329   Bias/Offset: 0.20640680047766577
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4101]: 0.00086
    [36m[1m│ [22m[39m        Avg displacement in data: 0.0293
    [36m[1m└ [22m[39m        Weight/Scale: 0.7370393600657519   Bias/Offset: 0.20707503050527667
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4151]: 0.00081
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02838
    [36m[1m└ [22m[39m        Weight/Scale: 0.7368518049508945   Bias/Offset: 0.20761702284552527
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4201]: 0.00081
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02845
    [36m[1m└ [22m[39m        Weight/Scale: 0.7366881128909374   Bias/Offset: 0.20821223601676878
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4251]: 0.00076
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02749
    [36m[1m└ [22m[39m        Weight/Scale: 0.7364895023616118   Bias/Offset: 0.208746369607349
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4301]: 0.00071
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02665
    [36m[1m└ [22m[39m        Weight/Scale: 0.7362927271029647   Bias/Offset: 0.2092500996788868
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4351]: 0.00067
    [36m[1m│ [22m[39m        Avg displacement in data: 0.0259
    [36m[1m└ [22m[39m        Weight/Scale: 0.7360921548788921   Bias/Offset: 0.20975006660235718
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4401]: 0.00064
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02525
    [36m[1m└ [22m[39m        Weight/Scale: 0.7358791966884161   Bias/Offset: 0.21023759779756476
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4451]: 0.00061
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02468
    [36m[1m└ [22m[39m        Weight/Scale: 0.7356561245657441   Bias/Offset: 0.21070847267353646
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4501]: 0.00058
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02417
    [36m[1m└ [22m[39m        Weight/Scale: 0.735421216015542   Bias/Offset: 0.21115856587346601
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 2/2  Epoch: 5/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4551]: 0.00057
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02396
    [36m[1m└ [22m[39m        Weight/Scale: 0.7351722155225242   Bias/Offset: 0.21158457235561778
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4601]: 0.00055
    [36m[1m│ [22m[39m        Avg displacement in data: 0.0235
    [36m[1m└ [22m[39m        Weight/Scale: 0.7348758533004653   Bias/Offset: 0.21194912487561213
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4651]: 0.00053
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02308
    [36m[1m└ [22m[39m        Weight/Scale: 0.7345848446458138   Bias/Offset: 0.21229538968966
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4701]: 0.00052
    [36m[1m│ [22m[39m        Avg displacement in data: 0.0227
    [36m[1m└ [22m[39m        Weight/Scale: 0.7343054508081289   Bias/Offset: 0.2126491265064766
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4751]: 0.0005
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02234
    [36m[1m└ [22m[39m        Weight/Scale: 0.7340231316635806   Bias/Offset: 0.21299931300463582
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4801]: 0.00048
    [36m[1m│ [22m[39m        Avg displacement in data: 0.022
    [36m[1m└ [22m[39m        Weight/Scale: 0.7337381088852366   Bias/Offset: 0.2133452388364815
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4851]: 0.00047
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02168
    [36m[1m└ [22m[39m        Weight/Scale: 0.7334511254968487   Bias/Offset: 0.21368730388256318
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4901]: 0.00046
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02138
    [36m[1m└ [22m[39m        Weight/Scale: 0.7331619987984307   Bias/Offset: 0.2140260326882961
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4951]: 0.00045
    [36m[1m│ [22m[39m        Avg displacement in data: 0.0211
    [36m[1m└ [22m[39m        Weight/Scale: 0.7328706734727711   Bias/Offset: 0.21436233071933172
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [5001]: 0.00043
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02083
    [36m[1m└ [22m[39m        Weight/Scale: 0.7325764672317332   Bias/Offset: 0.21469677694279848
    

    1322.133255 seconds (10.66 G allocations: 526.566 GiB, 7.90% gc time)
    


    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_11.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_12.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_13.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_14.svg)
    


    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mFriction model 1 mse: 14.780632206979849
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mFriction model 2 mse: 18.61551936889592
    


    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_16.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_17.svg)
    


Finally, the FMU is cleaned-up.


```julia
fmiUnload(simpleFMU)
```

### Summary

Based on the plots, it can be seen that the curves of the *realFMU* and the *neuralFMU* are very close. The *neuralFMU* is able to learn the friction and displacement model.

### Source

[1] Tobias Thummerer, Lars Mikelsons and Josef Kircher. 2021. **NeuralFMU: towards structural integration of FMUs into neural networks.** Martin Sjölund, Lena Buffoni, Adrian Pop and Lennart Ochel (Ed.). Proceedings of 14th Modelica Conference 2021, Linköping, Sweden, September 20-24, 2021. Linköping University Electronic Press, Linköping (Linköping Electronic Conference Proceedings ; 181), 297-306. [DOI: 10.3384/ecp21181297](https://doi.org/10.3384/ecp21181297)

