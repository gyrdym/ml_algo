import 'package:ml_algo/src/classifier/knn_classifier/_injector.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_factory.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_impl.dart';
import 'package:ml_algo/src/common/exception/outdated_json_schema_exception.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_algo/src/knn_solver/neigbour.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../helpers.dart';
import '../../mocks.dart';
import '../../mocks.mocks.dart';

void main() {
  group('KnnClassifierImpl', () {
    final classLabelPrefix = 'awesome class label';
    final targetColumnName = 'target';
    final classLabels = [1, 2, 3];
    final dtype = DType.float32;
    final solverMock = MockKnnSolver();
    final kernelMock = MockKernel();
    final k = 10;
    final distanceType = Distance.hamming;
    final kernelType = KernelType.epanechnikov;

    tearDown(() {
      reset(solverMock);
      reset(kernelMock);
      injector.clearAll();
      knnClassifierInjector.clearAll();
    });

    group('default constructor', () {
      setUp(() {
        when(solverMock.k).thenReturn(k);
        when(solverMock.distanceType).thenReturn(distanceType);
        when(kernelMock.type).thenReturn(kernelType);
      });

      tearDown(() {
        reset(solverMock);
        reset(kernelMock);
        injector.clearAll();
        knnClassifierInjector.clearAll();
      });

      test('should throw an exception if no class labels are provided', () {
        final classLabels = <num>[];
        final actual = () => KnnClassifierImpl(
          targetColumnName,
          classLabels,
          kernelMock,
          solverMock,
          classLabelPrefix,
          dtype,
        );

        expect(actual, throwsException);
      });

      test('should persist model hyperparameters', () {
        final classifier = KnnClassifierImpl(
          targetColumnName,
          classLabels,
          kernelMock,
          solverMock,
          classLabelPrefix,
          dtype,
        );

        expect(classifier.k, k);
        expect(classifier.kernelType, kernelType);
        expect(classifier.distanceType, distanceType);
      });
    });

    group('predict', () {
      final solverMock = MockKnnSolver();
      final kernelMock = MockKernel();

      setUp(() => when(kernelMock.getWeightByDistance(any, any)).thenReturn(1));

      tearDown(() {
        reset(solverMock);
        reset(kernelMock);
      });

      test('should throw an exception if no features are provided', () {
        final classifier = KnnClassifierImpl(
          targetColumnName,
          [1],
          kernelMock,
          solverMock,
          classLabelPrefix,
          dtype,
        );

        final features = DataFrame.fromMatrix(Matrix.empty());

        expect(() => classifier.predict(features), throwsException);
      });

      test('should return a dataframe with just one column, consisting of '
          'weighted majority-based outcomes of closest observations of provided '
          'features', () {
        final classifier = KnnClassifierImpl(
          targetColumnName,
          classLabels,
          kernelMock,
          solverMock,
          classLabelPrefix,
          dtype,
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
          targetColumnName,
          classLabels,
          kernelMock,
          solverMock,
          classLabelPrefix,
          dtype,
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

        expect(actual.header, equals([targetColumnName]));
      });

      test('should return a label of first neighbour among found k neighbours '
          'if there is no major class', () {
        final classifier = KnnClassifierImpl(
          targetColumnName,
          classLabels,
          kernelMock,
          solverMock,
          classLabelPrefix,
          dtype,
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
        final classifier = KnnClassifierImpl(
          targetColumnName,
          classLabels,
          kernelMock,
          solverMock,
          classLabelPrefix,
          dtype,
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
      final solverMock = MockKnnSolver();
      final kernelMock = MockKernel();

      setUp(() => when(kernelMock.getWeightByDistance(any, any))
          .thenReturn(1));

      tearDown(() {
        reset(solverMock);
        reset(kernelMock);
      });

      test('should return probability distribution of classes for each feature '
          'row', () {
        final classifier = KnnClassifierImpl(
          targetColumnName,
          classLabels,
          kernelMock,
          solverMock,
          classLabelPrefix,
          dtype,
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

        final expectedProbabilities = [
          [ 10 / 35,  15 / 35,  10 / 35 ],
          [ 11 / 41,  15 / 41,  15 / 41 ],
          [  5 / 11,   5 / 11,   1 / 11 ],
        ];

        expect(actual.rows, iterable2dAlmostEqualTo(expectedProbabilities));
      });

      test('should return probability distribution of classes where '
          'probabilities of absent class labels are 0.0', () {
        final classifier = KnnClassifierImpl(
          targetColumnName,
          classLabels,
          kernelMock,
          solverMock,
          classLabelPrefix,
          dtype,
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
        final classifier = KnnClassifierImpl(
          targetColumnName,
          classLabels,
          kernelMock,
          solverMock,
          classLabelPrefix,
          dtype,
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
          equals([
            '$classLabelPrefix 1',
            '$classLabelPrefix 2',
            '$classLabelPrefix 3'
          ]),
        );
      });

      test('should consider initial order of column labels', () {
        final firstClassLabel = 1;
        final secondClassLabel = 2;
        final thirdClassLabel = 3;

        final classLabels = [thirdClassLabel, firstClassLabel, secondClassLabel];

        final classifier = KnnClassifierImpl(
          targetColumnName,
          classLabels,
          kernelMock,
          solverMock,
          classLabelPrefix,
          dtype,
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
            equals([
              '$classLabelPrefix 3',
              '$classLabelPrefix 1',
              '$classLabelPrefix 2',
            ]),
        );
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
          targetColumnName,
          classLabels,
          kernelMock,
          solverMock,
          classLabelPrefix,
          dtype,
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

    group('retrain', () {
      final retrainingData = DataFrame([
        [1, 20, 300, 400],
      ]);
      final classifierFactory = MockKnnClassifierFactory();
      final retrainedModelMock = MockKnnClassifier();
      final classifier = KnnClassifierImpl(
        targetColumnName,
        classLabels,
        kernelMock,
        solverMock,
        classLabelPrefix,
        dtype,
      );

      setUp(() {
        when(solverMock.k).thenReturn(k);
        when(solverMock.distanceType).thenReturn(distanceType);
        when(kernelMock.type).thenReturn(kernelType);
        when(
          classifierFactory.create(
            any,
            any,
            any,
            any,
            any,
            any,
            any,
          )
        ).thenReturn(retrainedModelMock);

        knnClassifierInjector
            .registerSingleton<KnnClassifierFactory>(
                () => classifierFactory);
      });

      tearDown(() {
        injector.clearAll();
        knnClassifierInjector.clearAll();
      });

      test('should call classifier factory while retraining the model', () {
        classifier.retrain(retrainingData);

        verify(classifierFactory.create(
          retrainingData,
          targetColumnName,
          k,
          kernelType,
          distanceType,
          classLabelPrefix,
          dtype,
        )).called(1);
      });

      test('should return a new instance for the retrained model', () {
        final retrainedModel = classifier.retrain(retrainingData);

        expect(retrainedModel, same(retrainedModelMock));
        expect(retrainedModel, isNot(same(classifier)));
      });

      test('should throw exception if the model schema is outdated, '
          'schemaVersion is null', () {
        final model = KnnClassifierImpl(
          targetColumnName,
          classLabels,
          kernelMock,
          solverMock,
          classLabelPrefix,
          dtype,
          schemaVersion: null,
        );

        expect(() => model.retrain(retrainingData),
            throwsA(isA<OutdatedJsonSchemaException>()));
      });
    });
  });
}
