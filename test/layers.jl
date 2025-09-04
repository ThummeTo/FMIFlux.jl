#
# Copyright (c) 2021 Tobias Thummerer, Lars Mikelsons
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

using Statistics: mean, std

shift = [1.0, 2.0, 3.0]
scale = [4.0, 5.0, 6.0]

input = [0.5, 1.2, 1.0]
inputArray = [[-3.0, -2.0, -1.0], [3.0, 2.0, 1.0], [1.0, 2.0, 3.0]]

### ShiftScale ###
s = ShiftScale(shift, scale)
@test s(input) == [6.0, 16.0, 24.0]

s = ShiftScale(inputArray)
@test s(input) == [1.25, -0.4, -0.5]

s = ShiftScale(inputArray; range = (-1,1))
for i = 1:length(inputArray)
    res = s(collect(inputArray[j][i] for j = 1:length(inputArray[i])))
    @test max(res...) <= 1
    @test min(res...) >= -1
end

s = ShiftScale(inputArray; range = (-2,2))
for i = 1:length(inputArray)
    res = s(collect(inputArray[j][i] for j = 1:length(inputArray[i])))
    @test max(res...) <= 2
    @test min(res...) >= -2
end

s = ShiftScale(inputArray; range = :Normalize)
# ToDo: Test for :NormalDistribution

p, re = Flux.destructure(Flux.Chain(s))
@test length(p) == 6

### ScaleShift ###
s = ScaleShift(scale, shift)
@test s(input) == [3.0, 8.0, 9.0]

s = ScaleShift(inputArray)
@test s(input) == [-3.0, 4.4, 4.0]

p = ShiftScale(inputArray)
s = ScaleShift(p)
for i = 1:length(inputArray)
    in = collect(inputArray[j][i] for j = 1:length(inputArray[i]))
    @test p(in) != in
    @test s(p(in)) == in
end

# ToDo: Add remaining layers
