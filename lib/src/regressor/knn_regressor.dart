import 'package:ml_algo/src/algorithms/knn.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/regressor/regressor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/norm.dart';
import 'package:ml_linalg/vector.dart';

class KNNRegressor implements Regressor {
  KNNRegressor({
    this.k,
    this.distanceType,
  });

  final Norm distanceType;
  final int k;

  Matrix _observations;
  Matrix _outcomes;

  @override
  void fit(Matrix observations, Matrix outcomes, {Matrix initialWeights,
    bool isDataNormalized}) {
    _observations = observations;
    _outcomes = outcomes;
  }

  @override
  Matrix predict(Matrix observations) => Matrix.fromColumns([
    Vector.from(_generateOutcomes(observations)),
  ]);

  Iterable<double> _generateOutcomes(Matrix observations) sync* {
    for (final kNeighbours in findKNeighbours(k, _observations, _outcomes,
        observations)) {
      yield kNeighbours
          .fold<double>(0, (sum, pair) => sum + pair.item1) / k;
    }
  }

  @override
  double test(Matrix features, Matrix origLabels, MetricType metric) => null;

  @override
  Vector get weights => null;
}
