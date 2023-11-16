#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

# ToDo: Quick-fixes until patch release SciMLSensitivity v0.7.XX
import FMISensitivity.SciMLSensitivity: FakeIntegrator, u_modified!
import FMISensitivity.SciMLSensitivity.DiffEqBase: set_u!
function u_modified!(::FakeIntegrator, ::Bool)
    return nothing
end
function set_u!(::FakeIntegrator, u)
    return nothing
end

import FMISensitivity.ReverseDiff: increment_deriv!
function increment_deriv!(t::AbstractArray{<:ReverseDiff.TrackedReal}, x::ReverseDiff.ZeroTangent, args...)
    return nothing
end