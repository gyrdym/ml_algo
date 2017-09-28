part of 'package:dart_ml/src/implementation.dart';

class _GradientCalculatorImpl implements GradientCalculator {
  List<Float32x4Vector> _weightsDeltaMatrix;
  double _weightDelta;
  TargetFunction _function;

  void init(int numberOfArguments, double argumentDelta, TargetFunction function) {
    _weightsDeltaMatrix = _generateArgumentsDeltaMatrix(argumentDelta, numberOfArguments);
    _weightDelta = argumentDelta;
    _function = function;
  }

  Float32x4Vector getGradient(Float32x4Vector k, Float32x4Vector x, double y) {
    return new Float32x4Vector.from(
        new List<double>.generate(k.length, (int i) => _partialDerivative(k, x, y, i)));
  }

  double _partialDerivative(Float32x4Vector k, Float32x4Vector x, double y, targetWeightPosition) {
    Float32x4Vector deltaK = _weightsDeltaMatrix[targetWeightPosition];
    return (_function(k + deltaK, x, y) - _function(k - deltaK, x, y)) / 2 / _weightDelta;
  }

  List<Float32x4Vector> _generateArgumentsDeltaMatrix(double increment, int length) {
    List<Float32x4Vector> matrix = new List<Float32x4Vector>(length);

    for (int i = 0; i < length; i++) {
      matrix[i] = new Float32x4Vector.from(new List<double>.generate(length, (int idx) => idx == i ? increment : 0.0));
    }

    return matrix;
  }
}
