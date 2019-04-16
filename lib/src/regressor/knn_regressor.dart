import 'package:ml_algo/src/algorithms/knn/knn.dart';
import 'package:ml_algo/src/metric/factory.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/regressor/non_parametric_regressor.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class KNNRegressor implements NoNParametricRegressor {
  KNNRegressor({
    this.k,
    this.distanceType = Distance.euclidean,
    this.solverFn = findKNeighbours,
  });

  final Distance distanceType;
  final int k;
  final FindKnnFn solverFn;

  Matrix _observations;
  Matrix _outcomes;

  Vector get _zeroVector => _cachedZeroVector ??= Vector.zero(
      _outcomes.columnsNum);
  Vector _cachedZeroVector;

  @override
  void fit(Matrix observations, Matrix outcomes, {Matrix initialWeights,
    bool isDataNormalized}) {
    _observations = observations;
    _outcomes = outcomes;
  }

  @override
  Matrix predict(Matrix observations) => Matrix.fromRows(
    _generateOutcomes(observations).toList(growable: false));

  Iterable<Vector> _generateOutcomes(Matrix observations) sync* {
    for (final kNeighbours in solverFn(k, _observations, _outcomes,
        observations, distance: distanceType)) {
      yield kNeighbours
          .fold<Vector>(_zeroVector, (sum, pair) => sum + pair.label) / k;
    }
  }

  @override
  double test(Matrix features, Matrix origLabels, MetricType metricType) {
    final metric = MetricFactory.createByType(metricType);
    final prediction = predict(features);
    return metric.getScore(prediction, origLabels);
  }

  @override
  Vector get weights => null;
}
