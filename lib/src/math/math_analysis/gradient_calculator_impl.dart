import 'package:dart_ml/src/math/math_analysis/gradient_calculator.dart';
import 'package:linalg/vector.dart';

class GradientCalculatorImpl implements GradientCalculator {
  List<SIMDVector> _argumentDeltaMatrix;
  double _argumentDelta;

  @override
  SIMDVector getGradient(
    OptimizationFunction function,
    SIMDVector targetVector,
    Iterable<SIMDVector> vectorArgs,
    Iterable<double> scalarArgs,
    double argumentDelta
  ) {
    if (argumentDelta != _argumentDelta) {
      _argumentDeltaMatrix = _generateArgumentsDeltaMatrix(argumentDelta, targetVector.length);
      _argumentDelta = argumentDelta;
    }
    final gradient = List<double>.generate(
      targetVector.length,
      (int position) => _partialDerivative(
        function,
        argumentDelta,
        targetVector,
        vectorArgs,
        scalarArgs,
        position
      ));
    return Float32x4VectorFactory.from(gradient);
  }

  double _partialDerivative(
    OptimizationFunction function,
    double argumentDelta,
    SIMDVector targetVector,
    Iterable<SIMDVector> vectorArgs,
    Iterable<double> scalarArgs,
    int targetArgPosition
  ) {
    final deltaK = _argumentDeltaMatrix[targetArgPosition];
    return (function(targetVector + deltaK, vectorArgs, scalarArgs) -
            function(targetVector - deltaK, vectorArgs, scalarArgs)) / 2 / argumentDelta;
  }

  List<SIMDVector> _generateArgumentsDeltaMatrix(double delta, int length) {
    final matrix = List<SIMDVector>(length);
    for (int i = 0; i < length; i++) {
      matrix[i] = Float32x4VectorFactory.from(List<double>.generate(length, (int idx) => idx == i ? delta : 0.0));
    }
    return matrix;
  }
}
