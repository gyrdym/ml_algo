import 'package:ml_algo/src/knn_kernel/cosine_kernel.dart';
import 'package:ml_algo/src/knn_kernel/epanechnikov_kernel.dart';
import 'package:ml_algo/src/knn_kernel/gaussian_kernel.dart';
import 'package:ml_algo/src/knn_kernel/kernel.dart';
import 'package:ml_algo/src/knn_kernel/kernel_factory.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_algo/src/knn_kernel/uniform_kernel.dart';

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
