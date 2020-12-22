import 'package:ml_algo/src/knn_kernel/kernel.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type_json_keys.dart';

class UniformKernel implements Kernel {
  const UniformKernel();

  @override
  final type = KernelType.uniform;

  @override
  num getWeightByDistance(num distance, [num bandwidth = 1]) =>
      distance.abs() <= bandwidth
          ? 1/2
          : 0;

  @override
  String toJson() => uniformKernelEncodedValue;
}
