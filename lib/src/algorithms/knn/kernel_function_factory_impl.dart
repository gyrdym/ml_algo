import 'package:ml_algo/src/algorithms/knn/kernel.dart';
import 'package:ml_algo/src/algorithms/knn/kernel_function_factory.dart';
import 'package:ml_algo/src/algorithms/knn/kernel_type.dart';

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
