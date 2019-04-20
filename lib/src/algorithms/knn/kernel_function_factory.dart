import 'package:ml_algo/src/algorithms/knn/kernel.dart';
import 'package:ml_algo/src/algorithms/knn/kernel_type.dart';

abstract class KernelFunctionFactory {
  KernelFn createByType(Kernel type);
}
