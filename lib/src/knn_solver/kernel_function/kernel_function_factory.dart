import 'package:ml_algo/src/knn_solver/kernel_function/kernel_function.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_type.dart';

abstract class KernelFunctionFactory {
  KernelFn createByType(Kernel type);
}
