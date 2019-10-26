import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_impl.dart';
import 'package:ml_algo/src/knn_solver/neigbour.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:ml_tech/unit_testing/matchers/iterable_2d_almost_equal_to.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('KnnClassifierImpl', () {
    group('constructor', () {
      final solverMock = KnnSolverMock();
      final kernelMock = KernelMock();

      tearDown(() {
        reset(solverMock);
        reset(kernelMock);
      });

      test('should throw an exception if no class labels are provided', () {
        final classLabels = <num>[];
        final actual = () => KnnClassifierImpl(
          'target',
          classLabels,
          kernelMock,
          solverMock,
          DType.float32,
        );

        expect(actual, throwsException);
      });
    });

    group('predict method', () {
      final solverMock = KnnSolverMock();
      final kernelMock = KernelMock();

      setUp(() => when(kernelMock.getWeightByDistance(any, any)).thenReturn(1));

      tearDown(() {
        reset(solverMock);
        reset(kernelMock);
      });

      test('should throw an exception if no features are provided', () {
        final classifier = KnnClassifierImpl(
          'target',
          [1],
          kernelMock,
          solverMock,
          DType.float32,
        );

        final features = DataFrame.fromMatrix(Matrix.empty());

        expect(() => classifier.predict(features), throwsException);
      });

      test('should return a dataframe with just one column, consisting of '
          'weighted majority-based outcomes of closest observations of provided '
          'features', () {
        final classLabels = [1, 2, 3];
        final classifier = KnnClassifierImpl(
          'target',
          classLabels,
          kernelMock,
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

        when(kernelMock.getWeightByDistance(1)).thenReturn(10);
        when(kernelMock.getWeightByDistance(20)).thenReturn(15);
        when(kernelMock.getWeightByDistance(21)).thenReturn(10);

        when(kernelMock.getWeightByDistance(33)).thenReturn(11);
        when(kernelMock.getWeightByDistance(44)).thenReturn(15);
        when(kernelMock.getWeightByDistance(93)).thenReturn(15);

        when(kernelMock.getWeightByDistance(-1)).thenReturn(5);
        when(kernelMock.getWeightByDistance(-30)).thenReturn(5);
        when(kernelMock.getWeightByDistance(-40)).thenReturn(1);

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
          kernelMock,
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

      test('should return a label of first neighbour among found k neighbours '
          'if there is no major class', () {
        final classLabels = [1, 2, 3];
        final classifier = KnnClassifierImpl(
          'target',
          classLabels,
          kernelMock,
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
      });

      test('should return a label of neighbours with bigger weights even if '
          'they are not the majority', () {
        final classLabels = [1, 2, 3];
        final classifier = KnnClassifierImpl(
          'target',
          classLabels,
          kernelMock,
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
            Neighbour(0, Vector.fromList([1])),
            Neighbour(2, Vector.fromList([2])),
            Neighbour(3, Vector.fromList([1])),
          ],
        ];

        when(kernelMock.getWeightByDistance(0)).thenReturn(1);
        when(kernelMock.getWeightByDistance(2)).thenReturn(100);
        when(kernelMock.getWeightByDistance(3)).thenReturn(5);

        when(solverMock.findKNeighbours(testFeatureMatrix))
            .thenReturn(mockedNeighbours);

        final actual = classifier.predict(testFeatures);

        final expectedOutcomes = [
          [2],
        ];

        expect(actual.rows, equals(expectedOutcomes));
      });
    });

    group('predictProbability', () {
      final solverMock = KnnSolverMock();
      final kernelMock = KernelMock();

      setUp(() => when(kernelMock.getWeightByDistance(any, any)).thenReturn(1));

      tearDown(() {
        reset(solverMock);
        reset(kernelMock);
      });

      test('should return probability distribution of classes for each feature '
          'row', () {
        final classLabels = [1, 2, 3];
        final classifier = KnnClassifierImpl(
          'target',
          classLabels,
          kernelMock,
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
            Neighbour(21, Vector.fromList([3])),
          ],
          [
            Neighbour(33, Vector.fromList([1])),
            Neighbour(44, Vector.fromList([3])),
            Neighbour(93, Vector.fromList([2])),
          ],
          [
            Neighbour(-1, Vector.fromList([2])),
            Neighbour(-30, Vector.fromList([1])),
            Neighbour(-40, Vector.fromList([3])),
          ],
        ];

        when(kernelMock.getWeightByDistance(1)).thenReturn(10);
        when(kernelMock.getWeightByDistance(20)).thenReturn(15);
        when(kernelMock.getWeightByDistance(21)).thenReturn(10);

        when(kernelMock.getWeightByDistance(33)).thenReturn(11);
        when(kernelMock.getWeightByDistance(44)).thenReturn(15);
        when(kernelMock.getWeightByDistance(93)).thenReturn(15);

        when(kernelMock.getWeightByDistance(-1)).thenReturn(5);
        when(kernelMock.getWeightByDistance(-30)).thenReturn(5);
        when(kernelMock.getWeightByDistance(-40)).thenReturn(1);

        when(solverMock.findKNeighbours(testFeatureMatrix))
            .thenReturn(mockedNeighbours);

        final actual = classifier.predictProbabilities(testFeatures);

        final expectedOutcomes = [
          [ 10 / 35,  15 / 35,  10 / 35 ],
          [ 11 / 41,  15 / 41,  15 / 41 ],
          [  5 / 11,  5 / 11,    1 / 11 ],
        ];

        expect(actual.rows, iterable2dAlmostEqualTo(expectedOutcomes));
      });

      test('should return probability distribution of classes where '
          'probabilities of absent class labels are 0.0', () {
        final classLabels = [1, 2, 3];
        final classifier = KnnClassifierImpl(
          'target',
          classLabels,
          kernelMock,
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
            Neighbour(1, Vector.fromList([2])),
            Neighbour(20, Vector.fromList([2])),
            Neighbour(21, Vector.fromList([1])),
          ],
          [
            Neighbour(1, Vector.fromList([3])),
            Neighbour(20, Vector.fromList([3])),
            Neighbour(21, Vector.fromList([3])),
          ],
        ];

        when(kernelMock.getWeightByDistance(1)).thenReturn(10);
        when(kernelMock.getWeightByDistance(20)).thenReturn(15);
        when(kernelMock.getWeightByDistance(21)).thenReturn(10);

        when(solverMock.findKNeighbours(testFeatureMatrix))
            .thenReturn(mockedNeighbours);

        final actual = classifier.predictProbabilities(testFeatures);

        final expectedProbabilities = [
          [ 10 / 35,  25 / 35,  0.0 ],
          [     0.0,      0.0,  1.0 ],
        ];

        expect(actual.rows, iterable2dAlmostEqualTo(expectedProbabilities));
      });

      test('should return a dataframe with a header, containing proper column '
          'names', () {
        final classLabels = [1, 2, 3];

        final classifier = KnnClassifierImpl(
          'target',
          classLabels,
          kernelMock,
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
          ],
        ];

        when(solverMock.findKNeighbours(testFeatureMatrix))
            .thenReturn(mockedNeighbours);

        final actual = classifier.predictProbabilities(testFeatures);

        expect(actual.header,
            equals(['Class label 1', 'Class label 2', 'Class label 3']));
      });

      test('should consider initial order of column labels', () {
        final firstClassLabel = 1;
        final secondClassLabel = 2;
        final thirdClassLabel = 3;

        final classLabels = [thirdClassLabel, firstClassLabel, secondClassLabel];

        final classifier = KnnClassifierImpl(
          'target',
          classLabels,
          kernelMock,
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
            Neighbour(1, Vector.fromList([firstClassLabel])),
            Neighbour(10, Vector.fromList([secondClassLabel])),
            Neighbour(20, Vector.fromList([thirdClassLabel])),
          ],
        ];

        when(solverMock.findKNeighbours(testFeatureMatrix))
            .thenReturn(mockedNeighbours);

        final firstClassWeight = 100;
        final secondClassWeight = 90;
        final thirdClassWeight = 70;

        when(kernelMock.getWeightByDistance(1)).thenReturn(firstClassWeight);
        when(kernelMock.getWeightByDistance(10)).thenReturn(secondClassWeight);
        when(kernelMock.getWeightByDistance(20)).thenReturn(thirdClassWeight);

        final actual = classifier.predictProbabilities(testFeatures);
        final predictedProbabilities = actual.rows;

        expect(actual.header,
            equals(['Class label 3', 'Class label 1', 'Class label 2']));
        expect(predictedProbabilities, iterable2dAlmostEqualTo([
          [thirdClassWeight / 260, firstClassWeight / 260, secondClassWeight / 260],
        ]));
      });

      test('should throw an exception if provided knn solver learned on wrong '
          'class labels', () {
        final firstClassLabel = 1;
        final secondClassLabel = 2;
        final thirdClassLabel = 3;

        final unexpectedClassLabel = 100;

        final classLabels = [thirdClassLabel, firstClassLabel, secondClassLabel];

        final classifier = KnnClassifierImpl(
          'target',
          classLabels,
          kernelMock,
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
            Neighbour(20, Vector.fromList([unexpectedClassLabel])),
          ],
        ];

        when(solverMock.findKNeighbours(testFeatureMatrix))
            .thenReturn(mockedNeighbours);

        final actual = () => classifier.predictProbabilities(testFeatures);

        expect(actual, throwsException);
      });
    });
  });
}
