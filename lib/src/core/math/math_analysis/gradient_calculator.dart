part of 'package:dart_ml/src/core/interface.dart';

typedef double TargetFunction(Float32x4Vector a, Float32x4Vector b, double c);

abstract class GradientCalculator {
  void init(int numberOfArguments, double argumentDelta, TargetFunction function);
  Float32x4Vector getGradient(Float32x4Vector k, Float32x4Vector x, double y);
}
