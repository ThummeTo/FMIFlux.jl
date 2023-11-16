#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

function is64(model::Flux.Chain)
    params = Flux.params(model)

    for i in 1:length(params)
        for j in 1:length(params[i])
            if !isa(params[i][j], Float64)
                return false
            end
        end
    end

    return true
end

function convert64(model::Flux.Chain)
    Flux.fmap(Flux.f64, model)
end
