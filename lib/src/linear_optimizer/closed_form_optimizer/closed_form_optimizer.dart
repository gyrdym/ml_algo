import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_linalg/inverse.dart';
import 'package:ml_linalg/matrix.dart';

class ClosedFormOptimizer implements LinearOptimizer {
  ClosedFormOptimizer(this._features, this._labels);

  final Matrix _features;
  final Matrix _labels;

  @override
  List<num> get costPerIteration => [];

  @override
  Matrix findExtrema(
      {Matrix? initialCoefficients,
      bool isMinimizingObjective = true,
      bool collectLearningData = false}) {
    final transposedFeatures = _features.transpose();

    return (transposedFeatures * _features).inverse(Inverse.LU) *
        transposedFeatures *
        _labels;
  }
}
