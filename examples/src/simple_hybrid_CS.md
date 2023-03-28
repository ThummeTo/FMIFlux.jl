# Creation and training of CS-NeuralFMUs
Tutorial by Johannes Stoljar, Tobias Thummerer

*Last edit: 03.01.2023*

## License


```julia
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons, Johannes Stoljar
# Licensed under the MIT license. 
# See LICENSE (https://github.com/thummeto/FMIFlux.jl/blob/main/LICENSE) file in the project root for details.
```

## Motivation
The Julia Package *FMIFlux.jl* is motivated by the application of hybrid modeling. This package enables the user to integrate his simulation model between neural networks (NeuralFMU). For this, the simulation model must be exported as FMU (functional mock-up unit), which corresponds to a widely used standard. The big advantage of hybrid modeling with artificial neural networks is, that effects that are difficult to model (because they might be unknown) can be easily learned by the neural networks. For this purpose, the NeuralFMU is trained with measurement data containing the not modeled physical effect. The final product is a simulation model including the originally not modeled effects. Another big advantage of the NeuralFMU is that it works with little data, because the FMU already contains the characteristic functionality of the simulation and only the missing effects are added.

NeuralFMUs do not need to be as easy as in this example. Basically a NeuralFMU can combine different ANN topologies that manipulate any FMU-input (system state, system inputs, time) and any FMU-output (system state derivative, system outputs, other system variables). However, for this example a NeuralFMU topology as shown in the following picture is used.

![CS-NeuralFMU.svg](https://github.com/thummeto/FMIFlux.jl/blob/main/docs/src/examples/img/CSNeuralFMU.svg?raw=true)

*NeuralFMU (CS) from* [[1]](#Source).

## Introduction to the example
In this example, the model of a one-dimensional spring pendulum (with an external acting force) is used to learn the initial states. For this purpose, on the one hand the initial position of the mass of the pendulum is shifted and on the other hand the default position of the mass from the model is used. The model with the shifted initial position serves as reference and is called *referenceFMU* in the following. The model with the default position is further referenced with *defaultFMU*. At the beginning, the actual state of both simulations is shown, whereby clear deviations can be seen in the graphs. Afterwards, the *defaultFMU* is integrated into a co-simulation NeuralFMU (CS-NeuralFMU) architecture. By training the NeuralFMU, an attempt is made to learn the initial displacement of the *referenceFMU*. It can be clearly seen that the NeuralFMU learns this shift well in just a few training steps. 


## Target group
The example is primarily intended for users who work in the field of first principle and/or hybrid modeling and are further interested in hybrid model building. The example wants to show how simple it is to combine FMUs with machine learning and to illustrate the advantages of this approach.


## Other formats
Besides, this [Jupyter Notebook](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/simple_hybrid_CS.ipynb) there is also a [Julia file](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/simple_hybrid_CS.jl) with the same name, which contains only the code cells and for the documentation there is a [Markdown file](https://github.com/thummeto/FMIFlux.jl/blob/examples/examples/src/simple_hybrid_CS.md) corresponding to the notebook.  


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
using FMIZoo
using DifferentialEquations: Tsit5
import Plots

# set seed
import Random
Random.seed!(1234);
```

After importing the packages, the path to the *Functional Mock-up Units* (FMUs) is set. The FMU is a model exported meeting the *Functional Mock-up Interface* (FMI) Standard. The FMI is a free standard ([fmi-standard.org](http://fmi-standard.org/)) that defines a container and an interface to exchange dynamic models using a combination of XML files, binaries and C code zipped into a single file. 

The objec-orientated structure of the *SpringPendulumExtForce1D* can be seen in the following graphic. This model is a simple spring pendulum without friction, but with an external force. 

![svg](https://github.com/thummeto/FMIFlux.jl/blob/main/docs/src/examples/img/SpringPendulumExtForce1D.svg?raw=true)

Next, the start time and end time of the simulation are set. Finally, a step size is specified to store the results of the simulation at these time steps.


```julia
tStart = 0.0
tStep = 0.01
tStop = 5.0
tSave = tStart:tStep:tStop
```

### ReferenceFMU

In the next lines of code the FMU of the *referenceFMU* model is loaded from *FMIZoo.jl* and the information about the FMU is shown.  


```julia
referenceFMU = fmiLoad("SpringPendulumExtForce1D", "Dymola", "2022x")
fmiInfo(referenceFMU)
```

In the next steps the parameters are defined. The first parameter is the initial position of the mass, which is initilized with $1.3𝑚$. The second parameter is the initial velocity of the mass, which is initilized with $0\frac{m}{s}$. The FMU hase two states: The first state is the position of the mass and the second state is the velocity. In the function fmiSimulate() the *referenceFMU* is simulated, still specifying the start and end time, the parameters and which variables are recorded. After the simulation is finished the result of the *referenceFMU* can be plotted. This plot also serves as a reference for the later CS-NeuralFMU model.


```julia
param = Dict("mass_s0" => 1.3, "mass.v" => 0.0)   # increase amplitude, invert phase
vrs = ["mass.s", "mass.v", "mass.a"]
referenceSimData = fmiSimulate(referenceFMU, (tStart, tStop); parameters=param, recordValues=vrs, saveat=tSave)
fmiPlot(referenceSimData)
```

The data from the simulation of the *referenceFMU*, are divided into position, velocity and acceleration data. The data for the acceleration will be needed later. 


```julia
posReference = fmi2GetSolutionValue(referenceSimData, vrs[1])
velReference = fmi2GetSolutionValue(referenceSimData, vrs[2])
accReference = fmi2GetSolutionValue(referenceSimData, vrs[3])
```

### DefaultFMU

The following is a renaming for the *referenceFMU* to *defaultFMU*. The previous initial position of the mass is now set to the default position of the *defaultFMU*. The initial position of the mass is initilized with $0.5𝑚$ and initial velocity of the mass is initialized with $0\frac{m}{s}$.


```julia
defaultFMU = referenceFMU
param = Dict("mass_s0" => 0.5, "mass.v" => 0.0)
```

The following simulate and plot the *defaultFMU* just like the *referenceFMU*. The differences between both systems can be clearly seen from the plots. In the plots for the *defaultFMU* you can see that other oscillations occur due to the different starting positions. On the one hand the oscillation of the *defaultFMU* starts in the opposite direction of the *referenceFMU* and on the other hand the graphs for the velocity and acceleration differ clearly in the amplitude. In the following we try to learn the initial shift of the position so that the graphs for the acceleration of both graphs match.


```julia
defaultSimData = fmiSimulate(defaultFMU, (tStart, tStop); parameters=param, recordValues=vrs, saveat=tSave)
fmiPlot(defaultSimData)
```

The data from the simualtion of the *defaultFMU*, are divided into position, velocity and acceleration data. The data for the acceleration will be needed later.


```julia
posDefault = fmi2GetSolutionValue(defaultSimData, vrs[1])
velDefault = fmi2GetSolutionValue(defaultSimData, vrs[2])
accDefault = fmi2GetSolutionValue(defaultSimData, vrs[3])
```

## CS-NeuralFMU

In this section, the *defaultFMU* is inserted into a CS-NeuralFMU architecture. It has the goal to learn the initial state of the *referenceFMU*.


For the external force, a simple function is implemented that always returns a force of $0N$ at each time point. Also, all other functions and implementations would be possible here. Only for simplification reasons the function was chosen so simply.


```julia
function extForce(t)
    return [0.0]
end 
```

#### Loss function

In order to train our model, a loss function must be implemented. The solver of the NeuralFMU can calculate the gradient of the loss function. The gradient descent is needed to adjust the weights in the neural network so that the sum of the error is reduced and the model becomes more accurate.

The loss function in this implementation consists of the mean squared error (mse) from the acceleration data of the *referenceFMU* simulation (`accReference`) and the acceleration data of the network (`accNet`).
$$ e_{mse} = \frac{1}{n} \sum\limits_{i=0}^n (accReference[i] - accNet[i])^2 $$


```julia
# loss function for training
function lossSum(p)
    solution = csNeuralFMU(extForce, tStep; p=p)

    accNet = fmi2GetSolutionValue(solution, 1; isIndex=true)
    
    FMIFlux.Losses.mse(accReference, accNet)
end
```

#### Callback

To output the loss in certain time intervals, a callback is implemented as a function in the following. Here a counter is incremented, every twentieth pass the loss function is called and the average error is printed out.


```julia
# callback function for training
global counter = 0
function callb(p)
    global counter += 1

    if counter % 20 == 1
        avgLoss = lossSum(p[1])
        @info "Loss [$counter]: $(round(avgLoss, digits=5))"
    end
end
```

#### Structure of the CS-NeuralFMU

In the following, the topology of the CS-NeuralFMU is constructed. It consists of an input layer, which then leads into the *defaultFMU* model. The CS-FMU computes the outputs for the given system state and time step. After the *defaultFMU* follows a dense layer, which has exactly as many inputs as the model has outputs. The output of this layer consists of 16 output nodes and a *tanh* activation function. The next layer has 16 input and output nodes with the same activation function. The last layer is again a dense layer with 16 input nodes and the number of model outputs as output nodes. Here, it is important that no *tanh*-activation function follows, because otherwise the pendulums state values would be limited to the interval $[-1;1]$.


```julia
# NeuralFMU setup
numInputs = length(defaultFMU.modelDescription.inputValueReferences)
numOutputs = length(defaultFMU.modelDescription.outputValueReferences)

net = Chain(u -> defaultFMU(;u_refs=defaultFMU.modelDescription.inputValueReferences, u=u, y_refs=defaultFMU.modelDescription.outputValueReferences),
            Dense(numOutputs, 16, tanh),
            Dense(16, 16, tanh),
            Dense(16, numOutputs))
```

#### Definition of the CS-NeuralFMU

The instantiation of the CS-NeuralFMU is done as a one-liner. The FMU `defaultFMU`, the structure of the network `net`, start `tStart` and end time `tStop`, and the time steps `tSave` for saving are specified.


```julia
csNeuralFMU = CS_NeuralFMU(defaultFMU, net, (tStart, tStop); saveat=tSave);
```

#### Plot before training

Here the state trajectory of the *extForceFMU* is recorded. Doesn't really look like a pendulum yet, but the system is random initialized by default. In the plots later on, the effect of learning can be seen.


```julia
solutionBefore = csNeuralFMU(extForce, tStep)
accNeuralFMU = fmi2GetSolutionValue(solutionBefore, 1; isIndex=true)
Plots.plot(tSave, accNeuralFMU, label="acc CS-NeuralFMU", linewidth=2)
```

#### Training of the CS-NeuralFMU

For the training of the CS-NeuralFMU the parameters are extracted. The known Adam optimizer for minimizing the gradient descent is used as further passing parameters. In addition, the previously defined loss and callback function, as well as the number of epochs are passed.


```julia
# train
paramsNet = FMIFlux.params(csNeuralFMU)

optim = Adam()
FMIFlux.train!(lossSum, paramsNet, Iterators.repeated((), 1000), optim; cb=()->callb(paramsNet))
```

#### Comparison of the plots

Here three plots are compared with each other and only the acceleration of the mass is considered. The first plot represents the *defaultFMU*, the second represents the *referenceFMU* and the third plot represents the result after training the CS-NeuralFMU. 


```julia
# plot results mass.a
solutionAfter = csNeuralFMU(extForce, tStep)

fig = Plots.plot(xlabel="t [s]", ylabel="mass acceleration [m/s^2]", linewidth=2,
                 xtickfontsize=12, ytickfontsize=12,
                 xguidefontsize=12, yguidefontsize=12,
                 legendfontsize=8, legend=:topright)

accNeuralFMU = fmi2GetSolutionValue(solutionAfter, 1; isIndex=true)

Plots.plot!(fig, tSave, accDefault, label="defaultFMU", linewidth=2)
Plots.plot!(fig, tSave, accReference, label="referenceFMU", linewidth=2)
Plots.plot!(fig, tSave, accNeuralFMU, label="CS-NeuralFMU (1000 eps.)", linewidth=2)
fig 
```

Finally, the FMU is cleaned-up.


```julia
fmiUnload(defaultFMU)
```

### Summary

Based on the plots, it can be clearly seen that the CS-NeuralFMU model is able to learn the shift of the initial position. Even after only 1000 training steps, the curves overlap very much, so no further training with more runs is needed.

### Source

[1] Tobias Thummerer, Lars Mikelsons and Josef Kircher. 2021. **NeuralFMU: towards structural integration of FMUs into neural networks.** Martin Sjölund, Lena Buffoni, Adrian Pop and Lennart Ochel (Ed.). Proceedings of 14th Modelica Conference 2021, Linköping, Sweden, September 20-24, 2021. Linköping University Electronic Press, Linköping (Linköping Electronic Conference Proceedings ; 181), 297-306. [DOI: 10.3384/ecp21181297](https://doi.org/10.3384/ecp21181297)

