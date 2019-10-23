import 'dart:math' as math;

import 'package:ml_algo/src/knn_kernel/kernel.dart';

class CosineKernel implements Kernel {
  const CosineKernel();

  @override
  num getWeightByDistance(num distance, [num bandwidth = 1]) =>
      distance.abs() <= bandwidth
          ? math.pi / 4 * math.cos(math.pi / 2 * distance)
          : 0;
}
