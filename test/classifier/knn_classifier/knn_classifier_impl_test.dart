import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_impl.dart';
import 'package:ml_algo/src/knn_solver/neigbour.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('KnnClassifierImpl', () {
    group('constructor', () {
      final solverMock = KnnSolverMock();

      tearDown(() => reset(solverMock));

      test('should throw an exception if no class labels are provided', () {
        final classLabels = <num>[];
        final actual = () => KnnClassifierImpl(
          'target',
          classLabels,
          (_, [__]) => null,
          solverMock,
          DType.float32,
        );

        expect(actual, throwsException);
      });
    });

    group('predict method', () {
      final solverMock = KnnSolverMock();

      tearDown(() => reset(solverMock));

      test('should throw an exception if no features are provided', () {
        final classifier = KnnClassifierImpl(
          'target',
          [1],
          (_, [__]) => null,
          solverMock,
          DType.float32,
        );

        final features = DataFrame.fromMatrix(Matrix.empty());

        expect(() => classifier.predict(features), throwsException);
      });

      test('should return a dataframe with just one column, consisting of '
          'majority-based outcomes of closest observations of provided '
          'features', () {
        final classLabels = [1, 2, 3];
        final classifier = KnnClassifierImpl(
          'target',
          classLabels,
          (_, [__]) => 1,
          solverMock,
          DType.float32,
        );

        final testFeatureMatrix = Matrix.fromList(
          [
            [10, 10, 10, 10],
            [20, 20, 20, 20],
            [30, 30, 30, 30],
          ],
        );

        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);

        final mockedNeighbours = [
          [
            Neighbour(1, Vector.fromList([1])),
            Neighbour(20, Vector.fromList([2])),
            Neighbour(21, Vector.fromList([1])),
          ],
          [
            Neighbour(33, Vector.fromList([1])),
            Neighbour(44, Vector.fromList([3])),
            Neighbour(93, Vector.fromList([3])),
          ],
          [
            Neighbour(-1, Vector.fromList([2])),
            Neighbour(-30, Vector.fromList([2])),
            Neighbour(-40, Vector.fromList([1])),
          ],
        ];

        when(solverMock.findKNeighbours(testFeatureMatrix))
            .thenReturn(mockedNeighbours);

        final actual = classifier.predict(testFeatures);

        final expectedOutcomes = [
          [1],
          [3],
          [2],
        ];

        expect(actual.rows, equals(expectedOutcomes));
      });

      test('should return a dataframe, consisting of just one column with '
          'a proper name', () {
        final classLabels = [1, 2];

        final classifier = KnnClassifierImpl(
          'target',
          classLabels,
          (_, [__]) => 1,
          solverMock,
          DType.float32,
        );

        final testFeatureMatrix = Matrix.fromList(
          [
            [10, 10, 10, 10],
          ],
        );

        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);

        final mockedNeighbours = [
          [
            Neighbour(1, Vector.fromList([1])),
            Neighbour(20, Vector.fromList([2])),
            Neighbour(21, Vector.fromList([1])),
          ],
        ];

        when(solverMock.findKNeighbours(testFeatureMatrix))
            .thenReturn(mockedNeighbours);

        final actual = classifier.predict(testFeatures);

        expect(actual.header, equals(['target']));
      });

      test('should return a label of first neighbour among founf k neighbours '
          'if there is no major class', () {
        final classLabels = [1, 2, 3];
        final classifier = KnnClassifierImpl(
          'target',
          classLabels,
          (_, [__]) => 1,
          solverMock,
          DType.float32,
        );

        final testFeatureMatrix = Matrix.fromList(
          [
            [10, 10, 10, 10],
          ],
        );

        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);

        final mockedNeighbours = [
          [
            Neighbour(-1, Vector.fromList([3])),
            Neighbour(-30, Vector.fromList([2])),
            Neighbour(-40, Vector.fromList([1])),
          ],
        ];

        when(solverMock.findKNeighbours(testFeatureMatrix))
            .thenReturn(mockedNeighbours);

        final actual = classifier.predict(testFeatures);

        final expectedOutcomes = [
          [3],
        ];

        expect(actual.rows, equals(expectedOutcomes));
        expect(actual.header, equals(['target']));
      });
    });
  });
}
