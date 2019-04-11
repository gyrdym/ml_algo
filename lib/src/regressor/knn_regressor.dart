import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/regressor/regressor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/norm.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';
import 'package:tuple/tuple.dart';
import 'package:quiver/iterables.dart';

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
  Matrix predict(Matrix observations) {
    final allNeighbours = zip([_observations.rows, _outcomes.rows]);
    final firstKNeighbours = allNeighbours.take(5).toList(growable: false);
    final restNeighbours = allNeighbours.skip(5);
    for (final observation in observations.rows) {
      final firstKWithDistance = firstKNeighbours
          .map((pair) => Tuple2(pair.first.distanceTo(observation), pair.last))
          .toList(growable: false);
      restNeighbours.forEach((pair) {
        final distance = observation.distanceTo(pair.first);
        for (int l = k - 1; l >= 0; l--) {
          if (firstKWithDistance[l] > distance) {

          }
        }
      });
    }
    return null;
  }

  @override
  double test(Matrix features, Matrix origLabels, MetricType metric) {
    return null;
  }

  @override
  Vector get weights => null;
}
