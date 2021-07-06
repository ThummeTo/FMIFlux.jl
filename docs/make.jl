#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Documenter, FMIFlux

makedocs(sitename="FMIFlux.jl",
         pages=Any["Home" => "index.md"])

deploydocs(repo = "github.com/ThummeTo/FMIFlux.jl.git", devbranch = "main")
