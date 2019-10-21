import 'dart:math' as math;

import 'package:ml_algo/src/knn_solver/kernel_function/kernel.dart';

class GaussianKernel implements Kernel {
  const GaussianKernel();

  @override
  num getWeightByDistance(num distance, [num bandwidth = 1])  =>
      (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * distance * distance);
}
