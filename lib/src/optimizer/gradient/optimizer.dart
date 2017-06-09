import 'package:dart_ml/src/math/vector/vector.dart';
import 'package:dart_ml/src/optimizer/interface/optimizer.dart';
import 'package:dart_ml/src/optimizer/regularization/regularization.dart';

abstract class GradientOptimizer implements Optimizer {
  double _minWeightsDistance;
  double _learningRate;
  int _iterationLimit;
  Regularization _regularization;
  double _alpha;

  void configure(double learningRate, double minWeightsDistance, int iterationLimit, Regularization regularization,
                 {double alpha = .00001}) {

    if (minWeightsDistance == null && iterationLimit == null) {
      throw new Exception('You must specify at least one criterion of convergence');
    }

    _learningRate = learningRate;
    _minWeightsDistance = minWeightsDistance;
    _iterationLimit = iterationLimit;
    _regularization = regularization;
    _alpha = alpha;
  }

  Vector optimize(List<Vector> features, Vector labels, {Vector weights}) {
    weights = weights ?? new Vector.zero(features.first.length);
    double weightsDistance = double.MAX_FINITE;
    int iterationCounter = 0;

    while (weightsDistance > _minWeightsDistance && iterationCounter < _iterationLimit) {
      double eta = _learningRate / ++iterationCounter;
      Vector newWeights = _generateNewWeights(weights, features, labels, eta);
      weightsDistance = newWeights.distanceTo(weights);
      weights = newWeights;
    }

    return weights;
  }

  Iterable<int> getSampleRange(int totalSamplesCount);

  Vector _generateNewWeights(Vector weights, List<Vector> features, Vector labels, double eta) {
    Iterable<int> range = getSampleRange(features.length);

    int start = range.first;
    int end = range.last;

    List<Vector> featuresBatch = features.sublist(start, end);
    Vector labelsBatch = labels.cut(start, end);

    return _makeGradientStep(weights, featuresBatch, labelsBatch, eta);
  }

  Vector _makeGradientStep(Vector weights, List<Vector> data, Vector target, double eta) {
    Vector gradientSumVector = _calculateGradient(weights, data[0], target[0], eta);

    for (int i = 1; i < data.length; i++) {
      gradientSumVector += _calculateGradient(weights, data[i], target[i], eta);
    }

    return weights - gradientSumVector.scalarDiv(data.length * 1.0);
  }

  Vector _calculateGradient(Vector k, Vector x, double y, double eta) {
    Vector pureGradient = x.scalarMul(2.0 * eta).scalarMul(x.dot(k) - y);

    if (_regularization != null) {
      return pureGradient + _calcRegularizationVector(k);
    }

    return pureGradient;
  }

  Vector _calcRegularizationVector(Vector weights) {
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
