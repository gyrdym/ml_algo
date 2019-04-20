import 'dart:math' as math;

typedef KernelFn = double Function(double u);

double uniformKernel(double u) => 1;

double epanechnikovKernel(double u) => 0.75 * (1 - u * u);

double cosineKernel(double u) => math.pi / 4 * math.cos(math.pi / 2 * u);

double gaussianKernel(double u) => 1 / math.sqrt(2 * math.pi) *
    math.exp(-0.5 * u * u);
