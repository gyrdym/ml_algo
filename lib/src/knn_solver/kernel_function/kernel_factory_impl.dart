import 'package:ml_algo/src/knn_solver/kernel_function/cosine_kernel.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/epanechnikov_kernel.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/gaussian_kernel.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_factory.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_type.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/uniform_kernel.dart';

class KernelFactoryImpl implements KernelFactory {
  const KernelFactoryImpl();

  @override
  Kernel createByType(KernelType type) {
    switch (type) {
      case KernelType.uniform:
        return const UniformKernel();

      case KernelType.epanechnikov:
        return const EpanechnikovKernel();

      case KernelType.cosine:
        return const CosineKernel();

      case KernelType.gaussian:
        return const GaussianKernel();

      default:
        throw UnsupportedError('Unsupported kernel type - $type');
    }
  }
}
