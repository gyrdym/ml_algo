part of 'package:dart_ml/src/interface.dart';

abstract class DerivativeFinder {
  void configure(int numberOfArguments, double argumentDelta, ScoreFunction function, LossFunction metric);
  Float32x4Vector gradient(Float32x4Vector k, Float32x4Vector x, double y);
  double partialDerivative(Float32x4Vector k, Float32x4Vector deltaK, Float32x4Vector x, double y);
}
