
# FAQ

This list some common - often numerical - errors, that can be fixed by better understanding the ODE-Problem inside your FMU.

## Double callback crossing
### Description
Error message, a double zero-crossing happended, often during training a NeuralFMU.

### Example
- `Double callback crossing floating pointer reducer errored. Report this issue.`

### Reason
This could be, because the event inside of a NeuralFMU can't be located (often when using Zygote). 

### Fix
- Try to increase the root search interpolation points, this is computational expensive for FMUs with many events- and event-indicators. This can be done using `fmu.executionConfig.rootSearchInterpolationPoints = 100` (default value is `10`).