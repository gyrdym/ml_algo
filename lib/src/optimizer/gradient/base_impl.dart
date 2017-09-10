part of 'package:dart_ml/src/implementation.dart';

abstract class _GradientOptimizerImpl implements GradientOptimizer {
  final DerivativeFinder _derivativeFinder = injector.get(DerivativeFinder);

  //hyper parameters declaration
  double _minWeightsDistance;
  double _learningRate;
  int _iterationLimit;
  Regularization _regularization;
  double _alpha;
  double _argumentIncrement;
  //hyper parameters declaration end

  LossFunction _targetMetric;
  ScoreFunction _scoreFunction;

  void configure({double learningRate, double minWeightsDistance, int iterationLimit, Regularization regularization,
                 LossFunction lossFunction, ScoreFunction scoreFunction,
                 double alpha, double argumentIncrement}) {

    if (lossFunction == null || scoreFunction == null) {
      throw new Exception('You must specify both loss function and score function');
    }

    _learningRate = learningRate ?? 1e-5;
    _minWeightsDistance = minWeightsDistance ?? 1e-8;
    _iterationLimit = iterationLimit ?? 10000;
    _regularization = regularization ?? Regularization.L2;
    _alpha = alpha ?? 1e-5;
    _argumentIncrement = argumentIncrement ?? 1e-5;
    _targetMetric = lossFunction;
    _scoreFunction = scoreFunction;
  }

  Float32x4Vector findMinima(List<Float32x4Vector> features, Float32List labels, {Float32x4Vector weights}) {
    return _findExtrema(features, labels, weights: weights);
  }

  Float32x4Vector findMaxima(List<Float32x4Vector> features, Float32List labels, {Float32x4Vector weights}) {
    return _findExtrema(features, labels, weights: weights, findingMinima: false);
  }

  Float32x4Vector _findExtrema(List<Float32x4Vector> features, Float32List labels, {Float32x4Vector weights,
    bool findingMinima: true}) {

    weights = weights ?? new Float32x4Vector.zero(features.first.length);

    _derivativeFinder.configure(weights.length, _argumentIncrement, _scoreFunction, _targetMetric);

    double weightsDistance = double.MAX_FINITE;
    int iterationCounter = 0;

    while (weightsDistance > _minWeightsDistance && iterationCounter < _iterationLimit) {
      double eta = _learningRate / ++iterationCounter;
      Float32x4Vector newWeights = _generateNewWeights(weights, features, labels, eta, findingMinima: findingMinima);
      weightsDistance = newWeights.distanceTo(weights);
      weights = newWeights;
    }

    return weights;
  }

  Iterable<int> _getSamplesRange(int totalSamplesCount);

  Float32x4Vector _generateNewWeights(Float32x4Vector weights, List<Float32x4Vector> features, Float32List labels,
                                      double eta, {bool findingMinima: true}) {

    Iterable<int> range = _getSamplesRange(features.length);

    int start = range.first;
    int end = range.last;

    List<Float32x4Vector> featuresBatch = features.sublist(start, end);
    List<double> labelsBatch = labels.sublist(start, end);

    return _makeGradientStep(weights, featuresBatch, labelsBatch, eta, findingMinima: findingMinima);
  }

  Float32x4Vector _makeGradientStep(Float32x4Vector weights, List<Float32x4Vector> data, Float32List target, double eta,
                                    {bool findingMinima: true}) {

    Float32x4Vector gradientSumVector = _getExtendedGradient(weights, data[0], target[0]);

    for (int i = 1; i < data.length; i++) {
      gradientSumVector += _getExtendedGradient(weights, data[i], target[i]);
    }

    return findingMinima ?
           weights - gradientSumVector.scalarMul(eta / data.length)
            :
           weights + gradientSumVector.scalarMul(eta / data.length);
  }

  Float32x4Vector _getExtendedGradient(Float32x4Vector k, Float32x4Vector x, double y) {
    Float32x4Vector pureGradient = _derivativeFinder.gradient(k, x, y);

    if (_regularization != null) {
      return pureGradient + _calcRegularizationVector(k);
    }

    return pureGradient;
  }

  Float32x4Vector _calcRegularizationVector(Float32x4Vector weights) {
    switch (_regularization) {
      case Regularization.L1:
        return weights.scalarMul(0.0).scalarAdd(_alpha);

      case Regularization.L2:
        return weights.scalarMul(2.0 * _alpha);

      default:
        throw new UnimplementedError('Unimplemented regularization type $_regularization');
    }
  }
}
