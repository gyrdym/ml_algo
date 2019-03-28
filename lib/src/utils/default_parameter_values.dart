import 'dart:typed_data';

abstract class DefaultParameterValues {
  static const dtype = Float32x4;
  static const iterationsLimit = 100;
  static const minCoefficientsUpdate = 1e-12;
  static const initialLearningRate = 1e-3;
}
