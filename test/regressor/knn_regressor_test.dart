import 'package:ml_algo/src/algorithms/knn/neigbour.dart';
import 'package:ml_algo/src/regressor/knn_regressor.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() {
  group('KNNRegressor', () {
    test('should consider distance type', () {
      final solverFn = (int k, Matrix trainObservations, Matrix outcomes,
          Matrix observations, {Distance distance}) {
        expect(distance, Distance.cosine);
        return <Iterable<Neighbour<Vector>>>[[
          Neighbour(1.0, Vector.from([1.0]))]];
      };

      KNNRegressor(k: 2, distance: Distance.cosine, solverFn: solverFn)
        ..fit(Matrix.from([[1.0]]), Matrix.from([[1.0]]))
        ..predict(Matrix.from([[1.0]]));
    });
  });
}
