import 'package:ml_linalg/dtype.dart';

abstract class DefaultParameterValues {
  static const dtype = DType.float32;
  static const iterationsLimit = 100;
  static const minCoefficientsUpdate = 1e-12;
  static const initialLearningRate = 1e-3;
}
