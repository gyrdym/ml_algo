import 'package:ml_algo/src/knn_kernel/kernel.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type_json_keys.dart';

class EpanechnikovKernel implements Kernel {
  const EpanechnikovKernel();

  @override
  final type = KernelType.epanechnikov;

  @override
  num getWeightByDistance(num distance, [num bandwidth = 1]) =>
      distance.abs() <= bandwidth ? 0.75 * (1 - distance * distance) : 0;

  @override
  String toJson() => epanechnikovKernelEncodedValue;
}
