import 'dart:math' as math;

import 'package:ml_algo/src/knn_kernel/kernel.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type_json_keys.dart';

class GaussianKernel implements Kernel {
  const GaussianKernel();

  @override
  final type = KernelType.gaussian;

  @override
  num getWeightByDistance(num distance, [num bandwidth = 1])  =>
      (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * distance * distance);

  @override
  String toJson() => gaussianKernelEncodedValue;
}
