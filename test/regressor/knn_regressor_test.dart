import 'package:ml_algo/src/algorithms/knn/neigbour.dart';
import 'package:ml_algo/src/regressor/knn_regressor.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() {
  group('KNNRegressor', () {
    test('should consider k parameter', () {
      final solverFn = (int k, Matrix trainObservations, Matrix outcomes,
          Matrix observations, {Distance distance}) {
        expect(k, 2);
        return <Iterable<Neighbour<Vector>>>[[
          Neighbour(1.0, Vector.from([1.0]))]];
      };

      KNNRegressor(k: 2, distance: Distance.cosine, solverFn: solverFn)
        ..fit(Matrix.from([[1.0]]), Matrix.from([[1.0]]))
        ..predict(Matrix.from([[1.0]]));
    });

    test('should pass train observations to the solver function', () {
      final solverFn = (int k, Matrix trainObservations, Matrix outcomes,
          Matrix observations, {Distance distance}) {
        expect(trainObservations, equals([
          [1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
        ]));
        return <Iterable<Neighbour<Vector>>>[[
          Neighbour(1.0, Vector.from([1.0]))]];
      };

      KNNRegressor(k: 2, distance: Distance.cosine, solverFn: solverFn)
        ..fit(Matrix.from([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]), Matrix.from([
          [1.0],
          [2.0]
        ]))
        ..predict(Matrix.from([[10.0, 20.0, 30.0]]));
    });

    test('should pass train outcomes to the solver function', () {
      final solverFn = (int k, Matrix trainObservations, Matrix trainOutcomes,
          Matrix observations, {Distance distance}) {
        expect(trainOutcomes, equals([
          [1.0],
          [2.0]
        ]));
        return <Iterable<Neighbour<Vector>>>[[
          Neighbour(1.0, Vector.from([1.0]))]];
      };

      KNNRegressor(k: 2, distance: Distance.cosine, solverFn: solverFn)
        ..fit(Matrix.from([
          [1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
        ]), Matrix.from([
          [1.0],
          [2.0]
        ]))
        ..predict(Matrix.from([[10.0, 20.0, 30.0]]));
    });

    test('should pass observations to the solver function', () {
      final solverFn = (int k, Matrix trainObservations, Matrix trainOutcomes,
          Matrix observations, {Distance distance}) {
        expect(observations, equals([[10.0, 20.0, 30.0]]));
        return <Iterable<Neighbour<Vector>>>[[
          Neighbour(1.0, Vector.from([1.0]))]];
      };

      KNNRegressor(k: 2, distance: Distance.cosine, solverFn: solverFn)
        ..fit(Matrix.from([
          [1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
        ]), Matrix.from([
          [1.0],
          [2.0]
        ]))
        ..predict(Matrix.from([[10.0, 20.0, 30.0]]));
    });

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
