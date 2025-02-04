#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module LuxExt

using FMIFlux
using Lux

@warn "Lux extension is WIP!"

include(joinpath(@__DIR__, "..", "src", "Lux", "convert.jl"))
include(joinpath(@__DIR__, "..", "src", "Lux", "layers.jl"))

end # LuxExt
