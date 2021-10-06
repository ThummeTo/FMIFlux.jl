![FMIFlux.jl Logo](https://github.com/ThummeTo/FMIFlux.jl/blob/main/logo/dark/fmifluxjl_logo_640_320.png "FMIFlux.jl Logo")
# FMIFlux.jl

## What is FMIFlux.jl?
FMIFlux.jl is a free-to-use software library for the Julia programming language, which offers the ability to setup NeuralFMUs: You can place FMUs ([fmi-standard.org](http://fmi-standard.org/)) simply inside any feed-forward NN topology and still keep the resulting hybrid model trainable with a standard AD training process.

<!--- [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://ThummeTo.github.io/FMIFlux.jl/stable) --->
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://ThummeTo.github.io/FMIFlux.jl/dev) [![](https://github.com/ThummeTo/FMIFlux.jl/actions/workflows/Test.yml/badge.svg)]()

## How can I use FMIFlux.jl?
1. open a Julia-Command-Window, activate your prefered environemnt
1. goto package manager using ```]```
1. type ```add FMIFlux``` or ```add "https://github.com/ThummeTo/FMIFlux.jl"```
1. have a look in the ```example``` folder

## What is currently supported in FMIFlux.jl?
- building and training ME-NeuralFMUs with the default Flux-Front-End
- building and training CS-NeuralFMUs with the default Flux-Front-End
- ...

## What is under development in FMIFlux.jl?
- performance optimizations
- different modes for sensitivity estimation
- improoved documentation
- more examples
- ...

## What Platforms are supported?
FMI.jl is tested (and testing) under Julia Version 1.6 on Windows (latest) and Ubuntu (latest). Mac should work, but untested.

## How to cite? Related publications?
Tobias Thummerer, Lars Mikelsons and Josef Kircher. 2021. **NeuralFMU: towards structural integration of FMUs into neural networks** In Martin Sjölund, Lena Buffoni, Adrian Pop and Lennart Ochel (Ed.). Proceedings of 14th Modelica Conference 2021, Linköping, Sweden, September 20-24, 2021. Linköping University Electronic Press, Linköping (Linköping Electronic Conference Proceedings ; 181), 297-306. [DOI: 10.3384/ecp21181297](https://doi.org/10.3384/ecp21181297)

Tobias Thummerer, Johannes Tintenherr, Lars Mikelsons 2021 **Hybrid modeling of the human cardiovascular system using NeuralFMUs** (10th International Conference on Mathematical Modeling in Physical Sciences, Preprint, Accepted) [arXiv:2109.04880](https://arxiv.org/abs/2109.04880)