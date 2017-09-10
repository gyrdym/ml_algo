part of 'package:dart_ml/src/interface.dart';

typedef double TargetFunction(Float32x4Vector a, Float32x4Vector b, double c);

abstract class DerivativeFinder {
  void configure(int numberOfArguments, double argumentDelta, TargetFunction function);
  Float32x4Vector gradient(Float32x4Vector k, Float32x4Vector x, double y);
  double partialDerivative(Float32x4Vector k, Float32x4Vector deltaK, Float32x4Vector x, double y);
}
