import 'package:ml_algo/src/knn_solver/kernel_function/kernel.dart';
import 'package:ml_algo/src/knn_solver/kernel_function/kernel_type.dart';

abstract class KernelFactory {
  Kernel createByType(KernelType type);
}
