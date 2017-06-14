library gradient_optimizer_base;

import 'dart:math' as math show pow;
import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/math/vector/vector.dart';
import 'package:dart_ml/src/optimizer/regularization/regularization.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/base.dart';
import 'package:dart_ml/src/math/misc/randomizer/randomizer.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/batch.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/mini_batch.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/stochastic.dart';

part 'batch.dart';
part 'mini_batch.dart';
part 'stochastic.dart';

abstract class GradientOptimizerImpl implements GradientOptimizer {
  //hyper parameters declaration
  double _minWeightsDistance;
  double _learningRate;
  int _iterationLimit;
  Regularization _regularization;
  double _alpha;
  double _argumentIncrement;
  //hyper parameters declaration end

  Vector _increment;

  void configure(double learningRate, double minWeightsDistance, int iterationLimit, Regularization regularization,
                 {double alpha = .00001, double argumentIncrement = 1e-5}) {

    if (minWeightsDistance == null && iterationLimit == null) {
      throw new Exception('You must specify at least one criterion of convergence');
    }

    _learningRate = learningRate;
    _minWeightsDistance = minWeightsDistance;
    _iterationLimit = iterationLimit;
    _regularization = regularization;
    _alpha = alpha;
    _argumentIncrement = argumentIncrement;
  }

  Vector optimize(List<Vector> features, Vector labels, {Vector weights}) {
    weights = weights ?? new Vector.zero(features.first.length);
    _increment = new Vector.zero(weights.length);
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

  Iterable<int> _getSamplesRange(int totalSamplesCount);

  Vector _generateNewWeights(Vector weights, List<Vector> features, Vector labels, double eta) {
    Iterable<int> range = _getSamplesRange(features.length);

    int start = range.first;
    int end = range.last;

    List<Vector> featuresBatch = features.sublist(start, end);
    Vector labelsBatch = labels.cut(start, end);

    return _makeGradientStep(weights, featuresBatch, labelsBatch, eta);
  }

  Vector _makeGradientStep(Vector weights, List<Vector> data, Vector target, double eta) {
    Vector gradientSumVector = _meanGradient(weights, data[0], target[0]);

    for (int i = 1; i < data.length; i++) {
      gradientSumVector += _meanGradient(weights, data[i], target[i]);
    }

    return weights - gradientSumVector.scalarMul(eta / data.length);
  }

  SquaredLoss _targetMetric = new SquaredLoss();

  Vector _meanGradient(Vector k, Vector x, double y) {
    //x.scalarMul(2.0).scalarMul(x.dot(k) - y);
    Vector pureGradient = _gradient(k, x, y);

    if (_regularization != null) {
      return pureGradient + _calcRegularizationVector(k);
    }

    return pureGradient;
  }

  Vector _gradient(Vector k, Vector x, double y) {
    Vector gradient = new Vector(k.length);

    for (int i = 0; i < k.length; i++) {
      _increment[i] = _argumentIncrement;
      gradient[i] = _partialDerivative(k, x, y);
      _increment[i] = 0.0;
    }

    return gradient;
  }

  double _partialDerivative(Vector k, Vector x, double y) =>
      (_targetMetric.function(k + _increment, x, y) - _targetMetric.function(k - _increment, x, y)) / 2 / _argumentIncrement;

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

class SquaredLoss implements LossFunction {
  @override
  double function(Vector k, Vector x, double y) => math.pow(k.dot(x) - y, 2);
}

abstract class LossFunction {
  double function(Vector k, Vector x, double y);
}
