import 'package:ml_algo/src/knn_solver/kernel_function/kernel_function.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_function_factory.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_type.dart';

class KernelFunctionFactoryImpl implements KernelFunctionFactory {
  const KernelFunctionFactoryImpl();

  @override
  KernelFn createByType(Kernel type) {
    switch (type) {
      case Kernel.uniform:
        return uniformKernel;

      case Kernel.epanechnikov:
        return epanechnikovKernel;

      case Kernel.cosine:
        return cosineKernel;

      case Kernel.gaussian:
        return gaussianKernel;

      default:
        throw UnsupportedError('Unsupported kernel type - $type');
    }
  }
}
