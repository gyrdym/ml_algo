import 'package:ml_algo/src/knn_solver/kernel_function/kernel.dart';

class EpanechnikovKernel implements Kernel {
  const EpanechnikovKernel();

  @override
  num getWeightByDistance(num distance, [num bandwidth = 1]) =>
      distance.abs() <= bandwidth
          ? 0.75 * (1 - distance * distance)
          : 0;
}
