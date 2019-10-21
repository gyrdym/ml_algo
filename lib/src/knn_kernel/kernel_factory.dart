import 'package:ml_algo/src/knn_kernel/kernel.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';

abstract class KernelFactory {
  Kernel createByType(KernelType type);
}
