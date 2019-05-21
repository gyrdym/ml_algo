import 'package:ml_algo/src/algorithms/knn/neigbour.dart';
import 'package:ml_algo/src/regressor/knn_regressor.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() {
  final fakeKNeighbours = <Iterable<Neighbour<Vector>>>[[
    Neighbour(1.0, Vector.fromList([1.0]))]];

  group('KNNRegressor', () {
    test('should consider k parameter', () {
      final solverFn = (int k, Matrix trainObservations, Matrix outcomes,
          Matrix observations, {Distance distance}) {
        expect(k, 1);
        return fakeKNeighbours;
      };

      KNNRegressor(
          Matrix.fromList([[1.0]]),
          Matrix.fromList([[1.0]]),
          k: 1,
          distance: Distance.cosine,
          solverFn: solverFn
      )..predict(Matrix.fromList([[1.0]]));
    });

    test('should pass train observations to the solver function', () {
      final solverFn = (int k, Matrix trainObservations, Matrix outcomes,
          Matrix observations, {Distance distance}) {
        expect(trainObservations, equals([
          [1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
        ]));
        return fakeKNeighbours;
      };

      KNNRegressor(Matrix.fromList([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]), Matrix.fromList([
        [1.0],
        [2.0]
      ]), k: 2, distance: Distance.cosine, solverFn: solverFn)
        ..predict(Matrix.fromList([[10.0, 20.0, 30.0]]));
    });

    test('should pass train outcomes to the solver function', () {
      final solverFn = (int k, Matrix trainObservations, Matrix trainOutcomes,
          Matrix observations, {Distance distance}) {
        expect(trainOutcomes, equals([
          [1.0],
          [2.0]
        ]));
        return fakeKNeighbours;
      };

      KNNRegressor(Matrix.fromList([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]), Matrix.fromList([
        [1.0],
        [2.0]
      ]), k: 2, distance: Distance.cosine, solverFn: solverFn)
        ..predict(Matrix.fromList([[10.0, 20.0, 30.0]]));
    });

    test('should pass observations to the solver function', () {
      final solverFn = (int k, Matrix trainObservations, Matrix trainOutcomes,
          Matrix observations, {Distance distance}) {
        expect(observations, equals([[10.0, 20.0, 30.0]]));
        return fakeKNeighbours;
      };

      KNNRegressor(Matrix.fromList([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]), Matrix.fromList([
        [1.0],
        [2.0]
      ]), k: 2, distance: Distance.cosine, solverFn: solverFn)
        ..predict(Matrix.fromList([[10.0, 20.0, 30.0]]));
    });

    test('should consider distance type', () {
      final solverFn = (int k, Matrix trainObservations, Matrix outcomes,
          Matrix observations, {Distance distance}) {
        expect(distance, Distance.cosine);
        return fakeKNeighbours;
      };

      KNNRegressor(
          Matrix.fromList([[1.0]]),
          Matrix.fromList([[1.0]]),
          k: 1,
          distance: Distance.cosine,
          solverFn: solverFn)
        ..predict(Matrix.fromList([[1.0]]));
    });

    test('should throw an exception if number of training observations and '
        'number of training outcomes mismatch', () {
      final solverFn = (int k, Matrix trainObservations, Matrix outcomes,
          Matrix observations, {Distance distance}) => fakeKNeighbours;
      expect(
        () => KNNRegressor(Matrix.fromList([[1.0, 2,0]]), Matrix.fromList([
          [1.0],
          [3.0],
        ]), k: 1, distance: Distance.cosine, solverFn: solverFn),
        throwsException,
      );
    });

    test('should throw an exception if a value of k parameter and the number of'
        'training observations mismatch', () {
      final solverFn = (int k, Matrix trainObservations, Matrix outcomes,
          Matrix observations, {Distance distance}) => fakeKNeighbours;
      expect(
            () => KNNRegressor(Matrix.fromList([[1.0, 2,0]]), Matrix.fromList([
              [1.0],]), k: 3, distance: Distance.cosine, solverFn: solverFn),
        throwsException,
      );
    });
  });
}
