part of 'package:dart_ml/src/core/implementation.dart';

class _GradientCalculatorImpl implements GradientCalculator {
  List<Float32x4Vector> _argumentDeltaMatrix;
  double _argumentDelta;

  Float32x4Vector getGradient(
    OptimizationFunction function,
    Float32x4Vector targetVector,
    Iterable<Float32x4Vector> vectorArgs,
    Iterable<double> scalarArgs,
    double argumentDelta
  ) {
    if (argumentDelta != _argumentDelta) {
      _argumentDeltaMatrix = _generateArgumentsDeltaMatrix(argumentDelta, targetVector.length);
      _argumentDelta = argumentDelta;
    }
    final gradient = new List<double>.generate(
      targetVector.length,
      (int position) => _partialDerivative(
        function,
        argumentDelta,
        targetVector,
        vectorArgs,
        scalarArgs,
        position
      ));
    return new Float32x4Vector.from(gradient);
  }

  double _partialDerivative(
    OptimizationFunction function,
    double argumentDelta,
    Float32x4Vector targetVector,
    Iterable<Float32x4Vector> vectorArgs,
    Iterable<double> scalarArgs,
    targetArgPosition
  ) {
    final deltaK = _argumentDeltaMatrix[targetArgPosition];
    return (function(targetVector + deltaK, vectorArgs, scalarArgs) -
            function(targetVector - deltaK, vectorArgs, scalarArgs)) / 2 / argumentDelta;
  }

  List<Float32x4Vector> _generateArgumentsDeltaMatrix(double delta, int length) {
    final matrix = new List<Float32x4Vector>(length);
    for (int i = 0; i < length; i++) {
      matrix[i] = new Float32x4Vector.from(new List<double>.generate(length, (int idx) => idx == i ? delta : 0.0));
    }
    return matrix;
  }
}
