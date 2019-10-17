import 'dart:math' as math;

typedef KernelFn = num Function(num u, [num lambda]);

num uniformKernel(num u, [num lambda = 1]) => u.abs() <= lambda
    ? 1/2
    : 0;

num epanechnikovKernel(num u, [num lambda = 1]) => u.abs() <= lambda
    ? 0.75 * (1 - u * u)
    : 0;

num cosineKernel(num u, [num lambda = 1]) => u.abs() <= lambda
    ? math.pi / 4 * math.cos(math.pi / 2 * u)
    : 0;

num gaussianKernel(num u, [num lambda = 1]) =>
    (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * u * u);
