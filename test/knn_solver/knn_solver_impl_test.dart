import 'package:ml_algo/src/knn_solver/knn_solver_impl.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('KnnSolverImpl', () {
    group('constructor', () {
      test('should throw an exception if empty feature matrix is provided', () {
        final actual = () => KnnSolverImpl(
              Matrix.empty(dtype: DType.float32),
              Matrix.fromList([
                [1]
              ]),
              2,
              Distance.cosine,
              true,
            );

        expect(actual, throwsException);
      });

      test('should throw an exception if empty outcome matrix is provided', () {
        final actual = () => KnnSolverImpl(
              Matrix.fromList([
                [1, 1, 1]
              ]),
              Matrix.empty(dtype: DType.float32),
              1,
              Distance.cosine,
              true,
            );

        expect(actual, throwsException);
      });

      test(
          'should throw an exception if rows number of outcome '
          'matrix is greater than the rows number of feature matrix', () {
        final featureMatrix = Matrix.fromList([
          [1, 1, 1]
        ]);
        final outcomeMatrix = Matrix.fromList([
          [1],
          [2],
        ]);

        final actual = () => KnnSolverImpl(
              featureMatrix,
              outcomeMatrix,
              1,
              Distance.cosine,
              true,
            );

        expect(actual, throwsException);
      });

      test(
          'should throw an exception if rows number of outcome '
          'matrix is less than the rows number of feature matrix', () {
        final featureMatrix = Matrix.fromList([
          [1, 1, 1],
          [2, 2, 2],
        ]);
        final outcomeMatrix = Matrix.fromList([
          [1],
        ]);

        final actual = () => KnnSolverImpl(
              featureMatrix,
              outcomeMatrix,
              1,
              Distance.cosine,
              true,
            );

        expect(actual, throwsException);
      });

      test(
          'should throw an exception if outcome matrix is not a column '
          'matrix', () {
        final featureMatrix = Matrix.fromList([
          [1, 1, 1],
          [2, 2, 2],
        ]);
        final outcomeMatrix = Matrix.fromList([
          [1, 1],
          [2, 2],
        ]);

        final actual = () => KnnSolverImpl(
              featureMatrix,
              outcomeMatrix,
              1,
              Distance.cosine,
              true,
            );

        expect(actual, throwsException);
      });

      test(
          'should throw an exception if k parameter is greater than the number '
          'of rows of provided matrices', () {
        final actual = () => KnnSolverImpl(
              Matrix.fromList([
                [1, 1, 1, 1]
              ]),
              Matrix.fromList([
                [1]
              ]),
              2,
              Distance.cosine,
              true,
            );

        expect(actual, throwsRangeError);
      });

      test('should throw an exception if k parameter is less than 0', () {
        final actual = () => KnnSolverImpl(
              Matrix.fromList([
                [1, 1, 1, 1]
              ]),
              Matrix.fromList([
                [1]
              ]),
              -1,
              Distance.cosine,
              true,
            );

        expect(actual, throwsRangeError);
      });

      test('should throw an exception if k parameter is equal to 0', () {
        final actual = () => KnnSolverImpl(
              Matrix.fromList([
                [1, 1, 1, 1]
              ]),
              Matrix.fromList([
                [1]
              ]),
              0,
              Distance.cosine,
              true,
            );

        expect(actual, throwsRangeError);
      });
    });

    group('findKNeighbours method', () {
      test(
          'should throw an exception if the number of provided test feature '
          'matrix columns is less than the number of columns of train  feature '
          'matrix', () {
        final solver = KnnSolverImpl(
          Matrix.fromList([
            [1, 1, 1, 1],
            [2, 2, 2, 2],
          ]),
          Matrix.fromList([
            [1],
            [3],
          ]),
          1,
          Distance.cosine,
          true,
        );

        final testFeatures = Matrix.fromList([
          [10, 10, 10]
        ]);

        expect(() => solver.findKNeighbours(testFeatures), throwsException);
      });

      test(
          'should throw an exception if the number of provided test feature '
          'matrix columns is greater than the number of columns of train '
          'feature matrix', () {
        final solver = KnnSolverImpl(
          Matrix.fromList([
            [1, 1, 1, 1],
            [2, 2, 2, 2],
          ]),
          Matrix.fromList([
            [1],
            [3],
          ]),
          1,
          Distance.cosine,
          true,
        );

        final testFeatures = Matrix.fromList([
          [10, 10, 10, 100, 100]
        ]);

        expect(() => solver.findKNeighbours(testFeatures), throwsException);
      });

      test(
          'should find k neighbours for each of N passed observations, '
          '0 < k < N', () {
        final k = 3;

        final y1 = [100.0];
        final y2 = [200.0];
        final y3 = [300.0];
        final y4 = [400.0];
        final y5 = [500.0];
        final y6 = [600.0];
        final y7 = [700.0];
        final y8 = [800.0];

        final trainFeatures = Matrix.fromList([
          [15, 15, 15, 15, 15],
          [14, 14, 14, 14, 14],
          [16, 16, 16, 16, 16],
          [18, 18, 18, 18, 18],
          [17, 17, 17, 17, 17],
          [13, 13, 13, 13, 13],
          [12, 12, 12, 12, 12],
          [5, 5, 5, 5, 5],
        ]);

        final trainOutcomes = Matrix.fromList([y1, y2, y3, y4, y5, y6, y7, y8]);

        final testFeatures = Matrix.fromList([
          [10, 10, 10, 10, 10],
          [3, 3, 3, 3, 3],
        ]);

        final solver = KnnSolverImpl(
            trainFeatures, trainOutcomes, k, Distance.euclidean, true);

        final actual = solver.findKNeighbours(testFeatures).toList();

        expect(
          [
            actual[0].map((pair) => pair.label),
            actual[1].map((pair) => pair.label),
          ],
          equals([
            [y7, y6, y2],
            [y8, y7, y6],
          ]),
        );
      });

      test(
          'should find k neighbours for each of N passed observations, '
          'k = N', () {
        final k = 5;

        final y1 = [100.0];
        final y2 = [200.0];
        final y3 = [300.0];
        final y4 = [400.0];
        final y5 = [500.0];

        final trainFeatures = Matrix.fromList([
          [15, 15, 15, 15, 15],
          [14, 14, 14, 14, 14],
          [2, 2, 2, 2, 2],
          [18, 18, 18, 18, 18],
          [1, 1, 1, 1, 1],
        ]);

        final trainOutcomes = Matrix.fromList([y1, y2, y3, y4, y5]);

        final testFeatures = Matrix.fromList([
          [10, 10, 10, 10, 10],
          [3, 3, 3, 3, 3],
        ]);

        final solver = KnnSolverImpl(
            trainFeatures, trainOutcomes, k, Distance.euclidean, true);

        final actual = solver.findKNeighbours(testFeatures).toList();

        expect(
          [
            actual[0].map((pair) => pair.label),
            actual[1].map((pair) => pair.label),
          ],
          equals([
            [y2, y1, y3, y4, y5],
            [y3, y5, y2, y1, y4],
          ]),
        );
      });
    });
  });
}
