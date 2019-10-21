import 'package:ml_algo/src/knn_solver/kernel_function/kernel.dart';

class UniformKernel implements Kernel {
  const UniformKernel();

  @override
  num getWeightByDistance(num distance, [num bandwidth = 1]) =>
      distance.abs() <= bandwidth
          ? 1/2
          : 0;
}
