#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

module FluxExt

using FMIFlux
import Flux

include(joinpath(@__DIR__, "..", "src", "Flux", "convert.jl"))
include(joinpath(@__DIR__, "..", "src", "Flux", "misc.jl"))
include(joinpath(@__DIR__, "..", "src", "Flux", "layers.jl"))
include(joinpath(@__DIR__, "..", "src", "Flux", "overload.jl"))
include(joinpath(@__DIR__, "..", "src", "Flux", "optimiser.jl"))

end # FluxExt
