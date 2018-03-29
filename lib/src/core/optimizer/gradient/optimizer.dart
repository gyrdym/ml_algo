part of 'package:dart_ml/src/core/implementation.dart';

class _GradientOptimizerImpl implements Optimizer {
  final LossFunction _lossFunction = coreInjector.get(LossFunction);
  final ScoreFunction _scoreFunction = coreInjector.get(ScoreFunction);
  final GradientCalculator _gradientCalculator = coreInjector.get(GradientCalculator);
  final LearningRateGenerator _learningRateGenerator = coreInjector.get(LearningRateGenerator);
  final InitialWeightsGenerator _initialWeightsGenerator = coreInjector.get(InitialWeightsGenerator);

  //hyper parameters declaration
  final double _weightsDiffThreshold;
  final int _iterationLimit;
  final double _lambda;
  final double _argumentIncrement;
  //hyper parameters declaration end

  _GradientOptimizerImpl({
    double learningRate,
    double minWeightsDiff,
    int iterationLimit,
    double lambda,
    double argumentIncrement
  }) :
    _weightsDiffThreshold = minWeightsDiff ?? 1e-8,
    _iterationLimit = iterationLimit ?? 10000,
    _lambda = lambda ?? 1e-5,
    _argumentIncrement = argumentIncrement ?? 1e-5
  {
    _learningRateGenerator.init(learningRate ?? 1e-5);
  }

  Float32x4Vector findExtrema(
    List<Float32x4Vector> features,
    Float32List labels,
    {
      Float32x4Vector weights,
      bool isMinimizingObjective = true
    }
  ) {
    weights = weights ?? _initialWeightsGenerator.generate(features.first.length);
    double weightsDistance = double.MAX_FINITE;
    int iterationCounter = 0;

    while (weightsDistance > _weightsDiffThreshold && iterationCounter++ < _iterationLimit) {
      final eta = _learningRateGenerator.getNextValue();
      final newWeights = _generateNewWeights(weights, features, labels, eta, isMinimization: isMinimizingObjective);
      weightsDistance = newWeights.distanceTo(weights);
      weights = newWeights;
    }

    _learningRateGenerator.stop();

    return weights;
  }

  Iterable<int> _getBatchRange(int numberOfPoints) =>
      throw new UnimplementedError('it is necesssary to implement this method in a heir class');

  Float32x4Vector _generateNewWeights(
    Float32x4Vector weights,
    List<Float32x4Vector> features,
    Float32List labels,
    double eta,
    {bool isMinimization: true}
  ) {
    final range = _getBatchRange(features.length);
    final start = range.first;
    final end = range.last;
    final featuresBatch = features.sublist(start, end);
    final labelsBatch = labels.sublist(start, end);

    return _makeGradientStep(weights, featuresBatch, labelsBatch, eta, isMinimization: isMinimization);
  }

  Float32x4Vector _makeGradientStep(
    Float32x4Vector weights,
    List<Float32x4Vector> data,
    Float32List target,
    double eta,
    {bool isMinimization: true}
  ) {
    Float32x4Vector gradientSumVector = _getGradient(weights, data[0], target[0]);
    for (int i = 1; i < data.length; i++) {
      gradientSumVector += _getGradient(weights, data[i], target[i]);
    }
    return isMinimization ? weights - gradientSumVector.scalarMul(eta) : weights + gradientSumVector.scalarMul(eta);
  }

  Float32x4Vector _getGradient(
    Float32x4Vector k,
    Float32x4Vector x,
    double y
  ) {
    final gradient = _gradientCalculator.getGradient(
      (Float32x4Vector k, List<Float32x4Vector> vectorArgs, List<double> scalarArgs) {
        final x = vectorArgs[0];
        final y = scalarArgs[0];
        final lambda = scalarArgs[1];
        return _lossFunction.loss(_scoreFunction.score(k, x), y) + lambda * k.norm(Norm.EUCLIDEAN);
      }, k, [x], [y, _lambda], _argumentIncrement
    );

    return gradient;
  }
}
