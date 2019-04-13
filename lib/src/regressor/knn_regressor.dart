import 'package:ml_algo/src/algorithms/knn/knn.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/regressor/regressor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/norm.dart';
import 'package:ml_linalg/vector.dart';

class KNNRegressor implements Regressor {
  KNNRegressor({
    this.k,
    this.distanceType,
    this.solverFn = findKNeighbours,
  });

  final Norm distanceType;
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
        observations)) {
      yield kNeighbours
          .fold<Vector>(_zeroVector, (sum, pair) => sum + pair.label) / k;
    }
  }

  @override
  double test(Matrix features, Matrix origLabels, MetricType metric) => null;

  @override
  Vector get weights => null;
}
