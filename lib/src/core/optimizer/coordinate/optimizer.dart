part of 'package:dart_ml/src/core/implementation.dart';

class _CoordinateOptimizerImpl implements Optimizer {
  final ScoreFunction _scoreFunction = coreInjector.get(ScoreFunction);
  final InitialWeightsGenerator _initialWeightsGenerator = coreInjector.get(InitialWeightsGenerator);

  //hyper parameters declaration
  final double _weightsDiffThreshold;
  final int _iterationLimit;
  final double _lambda;
  //hyper parameters declaration end

  _CoordinateOptimizerImpl({
    double minWeightsDiff,
    int iterationLimit,
    double lambda
  }) :
    _weightsDiffThreshold = minWeightsDiff ?? 1e-8,
    _iterationLimit = iterationLimit ?? 10000,
    _lambda = lambda ?? 1e-5;

  @override
  Float32x4Vector findExtrema(
    List<Float32x4Vector> points,
    Float32List labels,
    {
      Float32x4Vector initialWeights,
      bool isMinimizingObjective = true
    }
  ) {
    Float32x4Vector weights = initialWeights ?? _initialWeightsGenerator.generate(points.first.length);

    double weightsDiff = double.INFINITY;
    int iteration = 0;

    while (weightsDiff > _weightsDiffThreshold && iteration < _iterationLimit) {
      final updatedWeights = new List<double>.filled(weights.length, 0.0, growable: false);

      for (int j = 0; j < weights.length; j++) {
        final weightsAsList = weights.asList();
        weightsAsList[j] = 0.0;

        for (int i = 0; i < points.length; i++) {
          final pointAsList = points[i].asList();
          final x = pointAsList[j];
          final y = labels[i];

          pointAsList[j] = 0.0;

          final weightsWithoutJ = new Float32x4Vector.from(weightsAsList);
          final pointWithoutJ = new Float32x4Vector.from(pointAsList);
          final yHat = _scoreFunction.score(weightsWithoutJ, pointWithoutJ);

          updatedWeights[j] += x * (y - yHat);
        }
      }

      final newWeights = new Float32x4Vector.from(updatedWeights);
      weightsDiff = newWeights.distanceTo(weights);
      weights = newWeights;

      iteration++;
    }

    return weights;
  }
}