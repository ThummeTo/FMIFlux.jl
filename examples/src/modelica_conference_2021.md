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

    [33m[1m┌ [22m[39m[33m[1mWarning: [22m[39mError requiring `Enzyme` from `LinearSolve`
    [33m[1m│ [22m[39m  exception =
    [33m[1m│ [22m[39m   LoadError: ArgumentError: Package LinearSolve does not have Enzyme in its dependencies:
    [33m[1m│ [22m[39m   - You may have a partially installed environment. Try `Pkg.instantiate()`
    [33m[1m│ [22m[39m     to ensure all packages in the environment are installed.
    [33m[1m│ [22m[39m   - Or, if you have LinearSolve checked out for development and have
    [33m[1m│ [22m[39m     added Enzyme as a dependency but haven't updated your primary
    [33m[1m│ [22m[39m     environment's manifest file, try `Pkg.resolve()`.
    [33m[1m│ [22m[39m   - Otherwise you may need to report an issue with LinearSolve
    [33m[1m│ [22m[39m   Stacktrace:
    [33m[1m│ [22m[39m     [1] [0m[1mmacro expansion[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90m.\[39m[90m[4mloading.jl:1167[24m[39m[90m [inlined][39m
    [33m[1m│ [22m[39m     [2] [0m[1mmacro expansion[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90m.\[39m[90m[4mlock.jl:223[24m[39m[90m [inlined][39m
    [33m[1m│ [22m[39m     [3] [0m[1mrequire[22m[0m[1m([22m[90minto[39m::[0mModule, [90mmod[39m::[0mSymbol[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90mBase[39m [90m.\[39m[90m[4mloading.jl:1144[24m[39m
    [33m[1m│ [22m[39m     [4] [0m[1minclude[22m[0m[1m([22m[90mmod[39m::[0mModule, [90m_path[39m::[0mString[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90mBase[39m [90m.\[39m[90m[4mBase.jl:419[24m[39m
    [33m[1m│ [22m[39m     [5] [0m[1minclude[22m[0m[1m([22m[90mx[39m::[0mString[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[35mLinearSolve[39m [90mC:\Users\runneradmin\.julia\packages\LinearSolve\qCLK7\src\[39m[90m[4mLinearSolve.jl:1[24m[39m
    [33m[1m│ [22m[39m     [6] [0m[1mmacro expansion[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90mC:\Users\runneradmin\.julia\packages\Requires\Z8rfN\src\[39m[90m[4mRequires.jl:40[24m[39m[90m [inlined][39m
    [33m[1m│ [22m[39m     [7] top-level scope
    [33m[1m│ [22m[39m   [90m    @ [39m[90mC:\Users\runneradmin\.julia\packages\LinearSolve\qCLK7\src\[39m[90m[4minit.jl:16[24m[39m
    [33m[1m│ [22m[39m     [8] [0m[1meval[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90m.\[39m[90m[4mboot.jl:368[24m[39m[90m [inlined][39m
    [33m[1m│ [22m[39m     [9] [0m[1meval[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90mC:\Users\runneradmin\.julia\packages\LinearSolve\qCLK7\src\[39m[90m[4mLinearSolve.jl:1[24m[39m[90m [inlined][39m
    [33m[1m│ [22m[39m    [10] [0m[1m(::LinearSolve.var"#88#97")[22m[0m[1m([22m[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[35mLinearSolve[39m [90mC:\Users\runneradmin\.julia\packages\Requires\Z8rfN\src\[39m[90m[4mrequire.jl:101[24m[39m
    [33m[1m│ [22m[39m    [11] [0m[1mmacro expansion[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90m[4mtiming.jl:382[24m[39m[90m [inlined][39m
    [33m[1m│ [22m[39m    [12] [0m[1merr[22m[0m[1m([22m[90mf[39m::[0mAny, [90mlistener[39m::[0mModule, [90mmodname[39m::[0mString, [90mfile[39m::[0mString, [90mline[39m::[0mAny[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[36mRequires[39m [90mC:\Users\runneradmin\.julia\packages\Requires\Z8rfN\src\[39m[90m[4mrequire.jl:47[24m[39m
    [33m[1m│ [22m[39m    [13] [0m[1m(::LinearSolve.var"#87#96")[22m[0m[1m([22m[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[35mLinearSolve[39m [90mC:\Users\runneradmin\.julia\packages\Requires\Z8rfN\src\[39m[90m[4mrequire.jl:100[24m[39m
    [33m[1m│ [22m[39m    [14] [0m[1mwithpath[22m[0m[1m([22m[90mf[39m::[0mAny, [90mpath[39m::[0mString[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[36mRequires[39m [90mC:\Users\runneradmin\.julia\packages\Requires\Z8rfN\src\[39m[90m[4mrequire.jl:37[24m[39m
    [33m[1m│ [22m[39m    [15] [0m[1m(::LinearSolve.var"#86#95")[22m[0m[1m([22m[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[35mLinearSolve[39m [90mC:\Users\runneradmin\.julia\packages\Requires\Z8rfN\src\[39m[90m[4mrequire.jl:99[24m[39m
    [33m[1m│ [22m[39m    [16] [0m[1m#invokelatest#2[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90m.\[39m[90m[4messentials.jl:729[24m[39m[90m [inlined][39m
    [33m[1m│ [22m[39m    [17] [0m[1minvokelatest[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90m.\[39m[90m[4messentials.jl:726[24m[39m[90m [inlined][39m
    [33m[1m│ [22m[39m    [18] [0m[1mforeach[22m[0m[1m([22m[90mf[39m::[0mtypeof(Base.invokelatest), [90mitr[39m::[0mVector[90m{Function}[39m[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90mBase[39m [90m.\[39m[90m[4mabstractarray.jl:2774[24m[39m
    [33m[1m│ [22m[39m    [19] [0m[1mloadpkg[22m[0m[1m([22m[90mpkg[39m::[0mBase.PkgId[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[36mRequires[39m [90mC:\Users\runneradmin\.julia\packages\Requires\Z8rfN\src\[39m[90m[4mrequire.jl:27[24m[39m
    [33m[1m│ [22m[39m    [20] [0m[1m#invokelatest#2[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90m.\[39m[90m[4messentials.jl:729[24m[39m[90m [inlined][39m
    [33m[1m│ [22m[39m    [21] [0m[1minvokelatest[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90m.\[39m[90m[4messentials.jl:726[24m[39m[90m [inlined][39m
    [33m[1m│ [22m[39m    [22] [0m[1mrun_package_callbacks[22m[0m[1m([22m[90mmodkey[39m::[0mBase.PkgId[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90mBase[39m [90m.\[39m[90m[4mloading.jl:869[24m[39m
    [33m[1m│ [22m[39m    [23] [0m[1m_tryrequire_from_serialized[22m[0m[1m([22m[90mmodkey[39m::[0mBase.PkgId, [90mpath[39m::[0mString, [90msourcepath[39m::[0mString, [90mdepmods[39m::[0mVector[90m{Any}[39m[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90mBase[39m [90m.\[39m[90m[4mloading.jl:944[24m[39m
    [33m[1m│ [22m[39m    [24] [0m[1m_require_search_from_serialized[22m[0m[1m([22m[90mpkg[39m::[0mBase.PkgId, [90msourcepath[39m::[0mString, [90mbuild_id[39m::[0mUInt64[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90mBase[39m [90m.\[39m[90m[4mloading.jl:1028[24m[39m
    [33m[1m│ [22m[39m    [25] [0m[1m_require[22m[0m[1m([22m[90mpkg[39m::[0mBase.PkgId[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90mBase[39m [90m.\[39m[90m[4mloading.jl:1315[24m[39m
    [33m[1m│ [22m[39m    [26] [0m[1m_require_prelocked[22m[0m[1m([22m[90muuidkey[39m::[0mBase.PkgId[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90mBase[39m [90m.\[39m[90m[4mloading.jl:1200[24m[39m
    [33m[1m│ [22m[39m    [27] [0m[1mmacro expansion[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90m.\[39m[90m[4mloading.jl:1180[24m[39m[90m [inlined][39m
    [33m[1m│ [22m[39m    [28] [0m[1mmacro expansion[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90m.\[39m[90m[4mlock.jl:223[24m[39m[90m [inlined][39m
    [33m[1m│ [22m[39m    [29] [0m[1mrequire[22m[0m[1m([22m[90minto[39m::[0mModule, [90mmod[39m::[0mSymbol[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90mBase[39m [90m.\[39m[90m[4mloading.jl:1144[24m[39m
    [33m[1m│ [22m[39m    [30] [0m[1meval[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90m.\[39m[90m[4mboot.jl:368[24m[39m[90m [inlined][39m
    [33m[1m│ [22m[39m    [31] [0m[1minclude_string[22m[0m[1m([22m[90mmapexpr[39m::[0mtypeof(REPL.softscope), [90mmod[39m::[0mModule, [90mcode[39m::[0mString, [90mfilename[39m::[0mString[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90mBase[39m [90m.\[39m[90m[4mloading.jl:1428[24m[39m
    [33m[1m│ [22m[39m    [32] [0m[1msoftscope_include_string[22m[0m[1m([22m[90mm[39m::[0mModule, [90mcode[39m::[0mString, [90mfilename[39m::[0mString[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[32mSoftGlobalScope[39m [90mC:\Users\runneradmin\.julia\packages\SoftGlobalScope\u4UzH\src\[39m[90m[4mSoftGlobalScope.jl:65[24m[39m
    [33m[1m│ [22m[39m    [33] [0m[1mexecute_request[22m[0m[1m([22m[90msocket[39m::[0mZMQ.Socket, [90mmsg[39m::[0mIJulia.Msg[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[33mIJulia[39m [90mC:\Users\runneradmin\.julia\packages\IJulia\Vo51o\src\[39m[90m[4mexecute_request.jl:67[24m[39m
    [33m[1m│ [22m[39m    [34] [0m[1m#invokelatest#2[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90m.\[39m[90m[4messentials.jl:729[24m[39m[90m [inlined][39m
    [33m[1m│ [22m[39m    [35] [0m[1minvokelatest[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[90m.\[39m[90m[4messentials.jl:726[24m[39m[90m [inlined][39m
    [33m[1m│ [22m[39m    [36] [0m[1meventloop[22m[0m[1m([22m[90msocket[39m::[0mZMQ.Socket[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[33mIJulia[39m [90mC:\Users\runneradmin\.julia\packages\IJulia\Vo51o\src\[39m[90m[4meventloop.jl:8[24m[39m
    [33m[1m│ [22m[39m    [37] [0m[1m(::IJulia.var"#15#18")[22m[0m[1m([22m[0m[1m)[22m
    [33m[1m│ [22m[39m   [90m    @ [39m[33mIJulia[39m [90m.\[39m[90m[4mtask.jl:484[24m[39m
    [33m[1m│ [22m[39m   in expression starting at C:\Users\runneradmin\.julia\packages\LinearSolve\qCLK7\ext\LinearSolveEnzymeExt.jl:1
    [33m[1m└ [22m[39m[90m@ Requires C:\Users\runneradmin\.julia\packages\Requires\Z8rfN\src\require.jl:51[39m
    

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
    [36m[1m└ [22m[39m        Weight/Scale: 0.6028853135038907   Bias/Offset: 0.048283734555565776
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [101]: 0.39139
    [36m[1m│ [22m[39m        Avg displacement in data: 0.62562
    [36m[1m└ [22m[39m        Weight/Scale: 0.6409651046124878   Bias/Offset: 0.08735843907070884
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [151]: 0.35731
    [36m[1m│ [22m[39m        Avg displacement in data: 0.59775
    [36m[1m└ [22m[39m        Weight/Scale: 0.6705316836418109   Bias/Offset: 0.1191972610680217
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [201]: 0.33753
    [36m[1m│ [22m[39m        Avg displacement in data: 0.58098
    [36m[1m└ [22m[39m        Weight/Scale: 0.6940825390789699   Bias/Offset: 0.1452517678229061
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [251]: 0.32536
    [36m[1m│ [22m[39m        Avg displacement in data: 0.5704
    [36m[1m└ [22m[39m        Weight/Scale: 0.712976426998822   Bias/Offset: 0.16638589817550334
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [301]: 0.31728
    [36m[1m│ [22m[39m        Avg displacement in data: 0.56327
    [36m[1m└ [22m[39m        Weight/Scale: 0.7281070585893763   Bias/Offset: 0.18318816015086484
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [351]: 0.31096
    [36m[1m│ [22m[39m        Avg displacement in data: 0.55763
    [36m[1m└ [22m[39m        Weight/Scale: 0.7401134619775888   Bias/Offset: 0.19607878334321727
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [401]: 0.30275
    [36m[1m│ [22m[39m        Avg displacement in data: 0.55023
    [36m[1m└ [22m[39m        Weight/Scale: 0.749488840949892   Bias/Offset: 0.20521930892596327
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [451]: 0.28769
    [36m[1m│ [22m[39m        Avg displacement in data: 0.53637
    [36m[1m└ [22m[39m        Weight/Scale: 0.7570800775291507   Bias/Offset: 0.21053145205955603
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [501]: 0.23876
    [36m[1m│ [22m[39m        Avg displacement in data: 0.48863
    [36m[1m└ [22m[39m        Weight/Scale: 0.7632569000603809   Bias/Offset: 0.21045772056968381
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 1/2  Epoch: 2/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [551]: 0.17541
    [36m[1m│ [22m[39m        Avg displacement in data: 0.41882
    [36m[1m└ [22m[39m        Weight/Scale: 0.771537143711889   Bias/Offset: 0.2144939947353995
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [601]: 0.03236
    [36m[1m│ [22m[39m        Avg displacement in data: 0.17988
    [36m[1m└ [22m[39m        Weight/Scale: 0.7893108519201213   Bias/Offset: 0.24187535588428918
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [651]: 0.02469
    [36m[1m│ [22m[39m        Avg displacement in data: 0.15714
    [36m[1m└ [22m[39m        Weight/Scale: 0.7845046150547389   Bias/Offset: 0.23235055858916054
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [701]: 0.0205
    [36m[1m│ [22m[39m        Avg displacement in data: 0.14316
    [36m[1m└ [22m[39m        Weight/Scale: 0.7814979448323592   Bias/Offset: 0.22744038338867362
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [751]: 0.01846
    [36m[1m│ [22m[39m        Avg displacement in data: 0.13587
    [36m[1m└ [22m[39m        Weight/Scale: 0.7788922673373014   Bias/Offset: 0.223557141675287
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [801]: 0.01772
    [36m[1m│ [22m[39m        Avg displacement in data: 0.13311
    [36m[1m└ [22m[39m        Weight/Scale: 0.7767620625857625   Bias/Offset: 0.22071943004139702
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [851]: 0.0165
    [36m[1m│ [22m[39m        Avg displacement in data: 0.12845
    [36m[1m└ [22m[39m        Weight/Scale: 0.7749710577480444   Bias/Offset: 0.21876176339928183
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [901]: 0.01597
    [36m[1m│ [22m[39m        Avg displacement in data: 0.12638
    [36m[1m└ [22m[39m        Weight/Scale: 0.7734249564858203   Bias/Offset: 0.21717926340160806
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [951]: 0.01551
    [36m[1m│ [22m[39m        Avg displacement in data: 0.12456
    [36m[1m└ [22m[39m        Weight/Scale: 0.7720094963053792   Bias/Offset: 0.215963825531383
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1001]: 0.01515
    [36m[1m│ [22m[39m        Avg displacement in data: 0.12309
    [36m[1m└ [22m[39m        Weight/Scale: 0.7702980017075355   Bias/Offset: 0.21436032914672545
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 1/2  Epoch: 3/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1051]: 0.01554
    [36m[1m│ [22m[39m        Avg displacement in data: 0.12466
    [36m[1m└ [22m[39m        Weight/Scale: 0.7688108750368833   Bias/Offset: 0.21311452331091071
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1101]: 0.0145
    [36m[1m│ [22m[39m        Avg displacement in data: 0.12043
    [36m[1m└ [22m[39m        Weight/Scale: 0.7672291467477739   Bias/Offset: 0.21198311304412734
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1151]: 0.01421
    [36m[1m│ [22m[39m        Avg displacement in data: 0.1192
    [36m[1m└ [22m[39m        Weight/Scale: 0.7657507423083704   Bias/Offset: 0.21090344072072703
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1201]: 0.01402
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11839
    [36m[1m└ [22m[39m        Weight/Scale: 0.764393061781348   Bias/Offset: 0.2101020727683478
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1251]: 0.01364
    [36m[1m│ [22m[39m        Avg displacement in data: 0.1168
    [36m[1m└ [22m[39m        Weight/Scale: 0.7628733903271255   Bias/Offset: 0.20905703721426055
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1301]: 0.01335
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11553
    [36m[1m└ [22m[39m        Weight/Scale: 0.7613299578067576   Bias/Offset: 0.20803804328493514
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1351]: 0.01316
    [36m[1m│ [22m[39m        Avg displacement in data: 0.1147
    [36m[1m└ [22m[39m        Weight/Scale: 0.759702764210813   Bias/Offset: 0.20693904080494996
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1401]: 0.01286
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11342
    [36m[1m└ [22m[39m        Weight/Scale: 0.7581192507212828   Bias/Offset: 0.20588485393546768
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1451]: 0.01254
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11198
    [36m[1m└ [22m[39m        Weight/Scale: 0.7565348993654795   Bias/Offset: 0.2047987770875348
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1501]: 0.0125
    [36m[1m│ [22m[39m        Avg displacement in data: 0.11181
    [36m[1m└ [22m[39m        Weight/Scale: 0.7549783579126909   Bias/Offset: 0.20377821401616789
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 1/2  Epoch: 4/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1551]: 0.01205
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10975
    [36m[1m└ [22m[39m        Weight/Scale: 0.7535203525464862   Bias/Offset: 0.20288773996011866
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1601]: 0.01184
    [36m[1m│ [22m[39m        Avg displacement in data: 0.1088
    [36m[1m└ [22m[39m        Weight/Scale: 0.7518829769750376   Bias/Offset: 0.2016949821832375
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1651]: 0.01155
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10749
    [36m[1m└ [22m[39m        Weight/Scale: 0.7503758951362424   Bias/Offset: 0.20058661641149053
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1701]: 0.01134
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10647
    [36m[1m└ [22m[39m        Weight/Scale: 0.7489291769460715   Bias/Offset: 0.19954547735584588
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1751]: 0.01125
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10606
    [36m[1m└ [22m[39m        Weight/Scale: 0.7475724681837941   Bias/Offset: 0.1986576860409276
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1801]: 0.01106
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10515
    [36m[1m└ [22m[39m        Weight/Scale: 0.7461446784636426   Bias/Offset: 0.19766436707886573
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1851]: 0.01094
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10459
    [36m[1m└ [22m[39m        Weight/Scale: 0.7447367958003863   Bias/Offset: 0.19665419664155853
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1901]: 0.01042
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10206
    [36m[1m└ [22m[39m        Weight/Scale: 0.7433872720908423   Bias/Offset: 0.19565197321923133
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [1951]: 0.01117
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10571
    [36m[1m└ [22m[39m        Weight/Scale: 0.7421796995885824   Bias/Offset: 0.1947709411473253
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2001]: 0.01019
    [36m[1m│ [22m[39m        Avg displacement in data: 0.10097
    [36m[1m└ [22m[39m        Weight/Scale: 0.7410850533020027   Bias/Offset: 0.19399216452536794
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 1/2  Epoch: 5/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2051]: 0.00992
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09958
    [36m[1m└ [22m[39m        Weight/Scale: 0.7397295724683568   Bias/Offset: 0.19284135282377163
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2101]: 0.00972
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09858
    [36m[1m└ [22m[39m        Weight/Scale: 0.7386685508938785   Bias/Offset: 0.1919940395710552
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2151]: 0.00967
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09836
    [36m[1m└ [22m[39m        Weight/Scale: 0.738123302044129   Bias/Offset: 0.19172744222490962
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2201]: 0.00927
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09628
    [36m[1m└ [22m[39m        Weight/Scale: 0.7366123498278988   Bias/Offset: 0.1903622705326613
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2251]: 0.00904
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09508
    [36m[1m└ [22m[39m        Weight/Scale: 0.7357373826972021   Bias/Offset: 0.1895301529012907
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2301]: 0.00878
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09372
    [36m[1m└ [22m[39m        Weight/Scale: 0.7349796077285281   Bias/Offset: 0.1887615731924067
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2351]: 0.00847
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09205
    [36m[1m└ [22m[39m        Weight/Scale: 0.7342028747761814   Bias/Offset: 0.18803118255057832
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2401]: 0.00837
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09151
    [36m[1m└ [22m[39m        Weight/Scale: 0.7333758539408395   Bias/Offset: 0.18748185147980334
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2451]: 0.00833
    [36m[1m│ [22m[39m        Avg displacement in data: 0.09126
    [36m[1m└ [22m[39m        Weight/Scale: 0.7322828117957689   Bias/Offset: 0.18677930199573034
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2501]: 0.00789
    [36m[1m│ [22m[39m        Avg displacement in data: 0.08884
    [36m[1m└ [22m[39m        Weight/Scale: 0.7308654092584684   Bias/Offset: 0.1857675409388825
    

    1732.509627 seconds (12.89 G allocations: 689.218 GiB, 4.12% gc time, 0.09% compilation time)
    


    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_2.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_3.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_4.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_5.svg)
    


    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mFriction model 1 mse: 14.828649508144359
    


    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_7.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_8.svg)
    


    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 2/2  Epoch: 1/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2551]: 0.00768
    [36m[1m│ [22m[39m        Avg displacement in data: 0.08764
    [36m[1m└ [22m[39m        Weight/Scale: 0.7299646610018445   Bias/Offset: 0.18525447032783415
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2601]: 0.00772
    [36m[1m│ [22m[39m        Avg displacement in data: 0.08789
    [36m[1m└ [22m[39m        Weight/Scale: 0.7289225960512343   Bias/Offset: 0.184795665975683
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2651]: 0.00698
    [36m[1m│ [22m[39m        Avg displacement in data: 0.08357
    [36m[1m└ [22m[39m        Weight/Scale: 0.7276980266610504   Bias/Offset: 0.18398743041549465
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2701]: 0.00629
    [36m[1m│ [22m[39m        Avg displacement in data: 0.07928
    [36m[1m└ [22m[39m        Weight/Scale: 0.7267246507889341   Bias/Offset: 0.1832009565156083
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2751]: 0.0056
    [36m[1m│ [22m[39m        Avg displacement in data: 0.0748
    [36m[1m└ [22m[39m        Weight/Scale: 0.7262639649434146   Bias/Offset: 0.1826394110334678
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2801]: 0.00463
    [36m[1m│ [22m[39m        Avg displacement in data: 0.06807
    [36m[1m└ [22m[39m        Weight/Scale: 0.7270385165627937   Bias/Offset: 0.18270249150732437
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2851]: 0.00385
    [36m[1m│ [22m[39m        Avg displacement in data: 0.06206
    [36m[1m└ [22m[39m        Weight/Scale: 0.7288481545543539   Bias/Offset: 0.1834198924134013
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2901]: 0.00351
    [36m[1m│ [22m[39m        Avg displacement in data: 0.05924
    [36m[1m└ [22m[39m        Weight/Scale: 0.7313923669815571   Bias/Offset: 0.18522575482960243
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [2951]: 0.00319
    [36m[1m│ [22m[39m        Avg displacement in data: 0.05645
    [36m[1m└ [22m[39m        Weight/Scale: 0.7333418959881465   Bias/Offset: 0.1869817102110683
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3001]: 0.00285
    [36m[1m│ [22m[39m        Avg displacement in data: 0.05339
    [36m[1m└ [22m[39m        Weight/Scale: 0.7345521168512857   Bias/Offset: 0.1883928532355987
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 2/2  Epoch: 2/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3051]: 0.00267
    [36m[1m│ [22m[39m        Avg displacement in data: 0.0517
    [36m[1m└ [22m[39m        Weight/Scale: 0.7356152593523101   Bias/Offset: 0.189972953821115
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3101]: 0.00256
    [36m[1m│ [22m[39m        Avg displacement in data: 0.05062
    [36m[1m└ [22m[39m        Weight/Scale: 0.7363225321689999   Bias/Offset: 0.19131721669861046
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3151]: 0.00235
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04852
    [36m[1m└ [22m[39m        Weight/Scale: 0.7369025519844008   Bias/Offset: 0.192597557469707
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3201]: 0.00229
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04781
    [36m[1m└ [22m[39m        Weight/Scale: 0.7373882708669022   Bias/Offset: 0.19386890136681667
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3251]: 0.00218
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04672
    [36m[1m└ [22m[39m        Weight/Scale: 0.737708030610579   Bias/Offset: 0.1949884301267994
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3301]: 0.00207
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04547
    [36m[1m└ [22m[39m        Weight/Scale: 0.7379611196523018   Bias/Offset: 0.1960335125526787
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3351]: 0.00183
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04279
    [36m[1m└ [22m[39m        Weight/Scale: 0.7382035826487292   Bias/Offset: 0.19703282409825226
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3401]: 0.00188
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04332
    [36m[1m└ [22m[39m        Weight/Scale: 0.7381856702596902   Bias/Offset: 0.19776052877441658
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3451]: 0.00173
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04154
    [36m[1m└ [22m[39m        Weight/Scale: 0.7383363128081194   Bias/Offset: 0.19871578165101242
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3501]: 0.00166
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04077
    [36m[1m└ [22m[39m        Weight/Scale: 0.7383652835791126   Bias/Offset: 0.19949508714451736
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 2/2  Epoch: 3/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3551]: 0.00214
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04625
    [36m[1m└ [22m[39m        Weight/Scale: 0.7381045961525304   Bias/Offset: 0.1999854231702132
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3601]: 0.00168
    [36m[1m│ [22m[39m        Avg displacement in data: 0.04096
    [36m[1m└ [22m[39m        Weight/Scale: 0.7379915435956476   Bias/Offset: 0.20035415076520696
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3651]: 0.00139
    [36m[1m│ [22m[39m        Avg displacement in data: 0.0373
    [36m[1m└ [22m[39m        Weight/Scale: 0.7379961148078367   Bias/Offset: 0.20100804308856599
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3701]: 0.0013
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03604
    [36m[1m└ [22m[39m        Weight/Scale: 0.7376086604801818   Bias/Offset: 0.20132270413504283
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3751]: 0.00124
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03521
    [36m[1m└ [22m[39m        Weight/Scale: 0.7374650956235459   Bias/Offset: 0.20187177010536797
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3801]: 0.00119
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03448
    [36m[1m└ [22m[39m        Weight/Scale: 0.7373156475117146   Bias/Offset: 0.20243021710990108
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3851]: 0.00113
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03368
    [36m[1m└ [22m[39m        Weight/Scale: 0.7371763177580166   Bias/Offset: 0.20295112015427297
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3901]: 0.00112
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03343
    [36m[1m└ [22m[39m        Weight/Scale: 0.7370818522439813   Bias/Offset: 0.20359628077519876
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [3951]: 0.00107
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03264
    [36m[1m└ [22m[39m        Weight/Scale: 0.7367670405587948   Bias/Offset: 0.2039542609279038
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4001]: 0.00097
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03118
    [36m[1m└ [22m[39m        Weight/Scale: 0.7366139241623194   Bias/Offset: 0.20442598344731044
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 2/2  Epoch: 4/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4051]: 0.00091
    [36m[1m│ [22m[39m        Avg displacement in data: 0.03009
    [36m[1m└ [22m[39m        Weight/Scale: 0.7365931055000093   Bias/Offset: 0.205020152691504
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4101]: 0.00088
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02971
    [36m[1m└ [22m[39m        Weight/Scale: 0.7365010833681165   Bias/Offset: 0.20564220294443886
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4151]: 0.00087
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02956
    [36m[1m└ [22m[39m        Weight/Scale: 0.7363071526576298   Bias/Offset: 0.20621032039400425
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4201]: 0.00083
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02879
    [36m[1m└ [22m[39m        Weight/Scale: 0.736082921294574   Bias/Offset: 0.20669274197993737
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4251]: 0.00079
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02802
    [36m[1m└ [22m[39m        Weight/Scale: 0.7358894363533028   Bias/Offset: 0.20719197927096758
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4301]: 0.00074
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02722
    [36m[1m└ [22m[39m        Weight/Scale: 0.7356964339245974   Bias/Offset: 0.20769558896272147
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4351]: 0.0007
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02646
    [36m[1m└ [22m[39m        Weight/Scale: 0.7354835665909074   Bias/Offset: 0.2081867225888874
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4401]: 0.00066
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02573
    [36m[1m└ [22m[39m        Weight/Scale: 0.7352529274767685   Bias/Offset: 0.2086605670266894
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4451]: 0.00063
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02509
    [36m[1m└ [22m[39m        Weight/Scale: 0.7350157195770892   Bias/Offset: 0.20912669866164424
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4501]: 0.0006
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02452
    [36m[1m└ [22m[39m        Weight/Scale: 0.7347812824145634   Bias/Offset: 0.209588583347291
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mRun: 2/2  Epoch: 5/5
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4551]: 0.00058
    [36m[1m│ [22m[39m        Avg displacement in data: 0.024
    [36m[1m└ [22m[39m        Weight/Scale: 0.7345468843685267   Bias/Offset: 0.21004311634799439
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4601]: 0.00055
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02355
    [36m[1m└ [22m[39m        Weight/Scale: 0.7343043214608982   Bias/Offset: 0.21048252099578224
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4651]: 0.00055
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02347
    [36m[1m└ [22m[39m        Weight/Scale: 0.7340053172602468   Bias/Offset: 0.21087753803574313
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4701]: 0.00053
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02304
    [36m[1m└ [22m[39m        Weight/Scale: 0.7337261539370772   Bias/Offset: 0.2112614944712673
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4751]: 0.00051
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02262
    [36m[1m└ [22m[39m        Weight/Scale: 0.7334351552074263   Bias/Offset: 0.21159459985634813
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4801]: 0.0005
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02227
    [36m[1m└ [22m[39m        Weight/Scale: 0.7331570956909034   Bias/Offset: 0.2119467898532257
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4851]: 0.00048
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02194
    [36m[1m└ [22m[39m        Weight/Scale: 0.7328788614701447   Bias/Offset: 0.21230001188401773
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4901]: 0.00047
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02164
    [36m[1m└ [22m[39m        Weight/Scale: 0.732597845073837   Bias/Offset: 0.2126507585240595
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [4951]: 0.00046
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02136
    [36m[1m└ [22m[39m        Weight/Scale: 0.7323139192728957   Bias/Offset: 0.21299809381012438
    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39m  Loss [5001]: 0.00044
    [36m[1m│ [22m[39m        Avg displacement in data: 0.02109
    [36m[1m└ [22m[39m        Weight/Scale: 0.7320279357049061   Bias/Offset: 0.21334245075436276
    

    1705.074368 seconds (12.51 G allocations: 670.616 GiB, 4.09% gc time)
    


    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_11.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_12.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_13.svg)
    



    
![svg](modelica_conference_2021_files/modelica_conference_2021_55_14.svg)
    


    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mFriction model 1 mse: 14.828649508144359
    [36m[1m[ [22m[39m[36m[1mInfo: [22m[39mFriction model 2 mse: 18.55980123377252
    


    
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

