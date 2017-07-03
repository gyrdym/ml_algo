library gradient_optimizer_base;

import 'dart:typed_data' show Float32List;
import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_vector/vector.dart';
import 'package:dart_ml/src/optimizer/regularization/regularization.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/base.dart';
import 'package:dart_ml/src/math/misc/randomizer/randomizer.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/batch.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/mini_batch.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/stochastic.dart';
import 'package:dart_ml/src/loss_function/loss_function.dart';
import 'package:dart_ml/src/score_function/score_function.dart';

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

  List<Float32x4Vector> _weightsDeltaMatrix;
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

  Float32x4Vector optimize(List<Float32x4Vector> features, Float32List labels, {Float32x4Vector weights}) {
    weights = weights ?? new Float32x4Vector.zero(features.first.length);
    _weightsDeltaMatrix = _generateWeightsDeltaMatrix(_argumentIncrement, weights.length);
    double weightsDistance = double.MAX_FINITE;
    int iterationCounter = 0;

    while (weightsDistance > _minWeightsDistance && iterationCounter < _iterationLimit) {
      double eta = _learningRate / ++iterationCounter;
      Float32x4Vector newWeights = _generateNewWeights(weights, features, labels, eta);
      weightsDistance = newWeights.distanceTo(weights);
      weights = newWeights;
    }

    return weights;
  }

  List<Float32x4Vector> _generateWeightsDeltaMatrix(double increment, int length) {
    List<Float32x4Vector> matrix = new List<Float32x4Vector>(length);

    for (int i = 0; i < length; i++) {
      matrix[i] = new Float32x4Vector.from(new List<double>.generate(length, (int idx) => idx == i ? increment : 0.0));
    }

    return matrix;
  }

  Iterable<int> _getSamplesRange(int totalSamplesCount);

  Float32x4Vector _generateNewWeights(Float32x4Vector weights, List<Float32x4Vector> features, Float32List labels, double eta) {
    Iterable<int> range = _getSamplesRange(features.length);

    int start = range.first;
    int end = range.last;

    List<Float32x4Vector> featuresBatch = features.sublist(start, end);
    List<double> labelsBatch = labels.sublist(start, end);

    return _makeGradientStep(weights, featuresBatch, labelsBatch, eta);
  }

  Float32x4Vector _makeGradientStep(Float32x4Vector weights, List<Float32x4Vector> data, Float32List target, double eta) {
    Float32x4Vector gradientSumVector = _extendedGradient(weights, data[0], target[0]);

    for (int i = 1; i < data.length; i++) {
      gradientSumVector += _extendedGradient(weights, data[i], target[i]);
    }

    return weights - gradientSumVector.scalarMul(eta / data.length);
  }

  Float32x4Vector _extendedGradient(Float32x4Vector k, Float32x4Vector x, double y) {
    Float32x4Vector pureGradient = _gradient(k, x, y);

    if (_regularization != null) {
      return pureGradient + _calcRegularizationVector(k);
    }

    return pureGradient;
  }

  Float32x4Vector _gradient(Float32x4Vector k, Float32x4Vector x, double y) =>
    new Float32x4Vector.from(new List<double>.generate(k.length, (int i) => _partialDerivative(k, _weightsDeltaMatrix[i], x, y)));

  double _partialDerivative(Float32x4Vector k, Float32x4Vector deltaK, Float32x4Vector x, double y) =>
      (_targetMetric.loss(_scoreFunction.score(k + deltaK, x), y) -
       _targetMetric.loss(_scoreFunction.score(k - deltaK, x), y)) / 2 / _argumentIncrement;

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
