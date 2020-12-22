import 'package:ml_algo/src/knn_kernel/kernel_type.dart';

abstract class Kernel {
  KernelType get type;
  num getWeightByDistance(num distance, [num bandwidth]);
  String toJson();
}
