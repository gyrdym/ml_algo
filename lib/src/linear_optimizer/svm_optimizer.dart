import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:quiver/iterables.dart';
import 'package:xrange/xrange.dart';

class SVMOptimizer implements LinearOptimizer {
  SVMOptimizer({
    required Matrix features,
    required Matrix labels,
    required num learningRate,
    required int iterationLimit,
    required DType dtype,
  })  : _features = features,
        _labels = labels,
        _learningRate = learningRate,
        _iterations = integers(0, iterationLimit),
        _dtype = dtype;

  final Matrix _features;
  final Matrix _labels;
  final num _learningRate;
  final Iterable<int> _iterations;
  final DType _dtype;

  @override
  // TODO: implement costPerIteration
  List<num> get costPerIteration => throw UnimplementedError();

  @override
  Matrix findExtrema(
      {Matrix? initialCoefficients,
      bool isMinimizingObjective = true,
      bool collectLearningData = false}) {
    var coefficients = initialCoefficients ??
        Matrix.column(List.filled(_features.first.length, 0), dtype: _dtype);

    for (final epochs in _iterations) {
      final predicted = _features * coefficients;
      final production = predicted.multiply(_labels);

      enumerate(production.columns.first).forEach((indexed) {
        if (indexed.value >= 1) {
          coefficients =
              coefficients - (coefficients * 2 * 1 / epochs) * _learningRate;
        } else {
          final val = Matrix.fromColumns(
              [_features[indexed.index] * _labels[indexed.index][0]],
              dtype: _dtype);

          coefficients = coefficients +
              (val - coefficients * 2 * 1 / epochs) * _learningRate;
        }
      });
    }

    return coefficients;
  }
}
