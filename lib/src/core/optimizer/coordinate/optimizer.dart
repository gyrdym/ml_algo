part of 'package:dart_ml/src/core/implementation.dart';

class _CoordinateOptimizerImpl implements Optimizer {
  final LossFunction _lossFunction = coreInjector.get(LossFunction);
  final ScoreFunction _scoreFunction = coreInjector.get(ScoreFunction);
  final GradientCalculator _gradientCalculator = coreInjector.get(GradientCalculator);
  final InitialWeightsGenerator _initialWeightsGenerator = coreInjector.get(InitialWeightsGenerator);

  //hyper parameters declaration
  final double _weightsDiffThreshold;
  final int _iterationLimit;
  final double _lambda;
  final double _argumentDelta;
  //hyper parameters declaration end

  _CoordinateOptimizerImpl({
    double learningRate,
    double minWeightsDiff,
    int iterationLimit,
    double lambda,
    double argumentIncrement
  }) :
    _weightsDiffThreshold = minWeightsDiff ?? 1e-8,
    _iterationLimit = iterationLimit ?? 10000,
    _lambda = lambda ?? 1e-5,
    _argumentDelta = argumentIncrement ?? 1e-5;

  @override
  Float32x4Vector findExtrema(
    List<Float32x4Vector> features,
    Float32List labels,
    {
      Float32x4Vector weights,
      bool isMinimizingObjective = true
    }
  ) {

  }
}