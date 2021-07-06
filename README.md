![FMIFlux.jl Logo](https://github.com/ThummeTo/FMIFlux.jl/blob/main/logo/fmifluxjl_logo_640_320.png "FMIFlux.jl Logo")
# FMIFlux.jl

## What is FMIFlux.jl?
FMIFlux.jl is a free-to-use software library for the Julia programming language, which offers the ability to setup NeuralFMUs: You can place FMUs ([fmi-standard.org](http://fmi-standard.org/)) simply inside any feed-forward NN topology and still keep the resulting hybrid model trainable with a standard AD training process.

<!--- [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://ThummeTo.github.io/FMIFlux.jl/stable) --->
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://ThummeTo.github.io/FMIFlux.jl/dev)

## How can I use FMIFlux.jl?
1. open a Julia-Command-Window, activate your prefered environemnt
1. goto package manager using ```]```
1. type ```add FMIFlux``` or ```add "https://github.com/ThummeTo/FMIFlux.jl"```
1. have a look in the ```example``` folder

## What is currently supported in FMIFlux.jl?
- building and training ME-NeuralFMUs with the default Flux-Front-End
- building and training CS-NeuralFMUs 
- ...

## What is under development in FMIFlux.jl?
- different modes for sensitivity estimation
- documentation
- more examples
- ...
