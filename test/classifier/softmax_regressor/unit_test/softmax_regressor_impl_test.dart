import 'package:ml_algo/src/classifier/softmax_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_impl.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../../helpers.dart';
import '../../../mocks.dart';
import '../../../mocks.mocks.dart';

void main() {
  group('SoftmaxRegressorImpl', () {
    final linkFunctionMock = MockLinkFunction();
    final optimizerType = LinearOptimizerType.gradient;
    final iterationsLimit = 3;
    final initialLearningRate = 0.75;
    final decay = 1.95;
    final minCoefficientsUpdate = 0.3;
    final lambda = 12.5;
    final regularizationType = RegularizationType.L2;
    final randomSeed = 144;
    final batchSize = 2;
    final isFittingDataNormalized = true;
    final learningRateType = LearningRateType.timeBased;
    final initialCoefficientsType = InitialCoefficientsType.zeroes;
    final initialCoefficients = Matrix.fromList([
      [13, 43, 55]
    ]);
    final fitIntercept = true;
    final interceptScale = 10.0;
    final positiveLabel = 1.0;
    final negativeLabel = -1.0;
    final costPerIteration = [10, -10, 20, 2.3];
    final retrainingData = DataFrame([
      [1, 2, -90, 100]
    ]);
    final retrainedModelMock = MockSoftmaxRegressor();
    final classifierFactory =
        createSoftmaxRegressorFactoryMock(retrainedModelMock);
    final dtype = DType.float32;
    final collectLearningData = false;

    final coefficientsByClasses = Matrix.fromList([
      [-1, -3, -5],
      [2, 10, 200],
      [1, 40, 300],
      [3, 44, -120],
      [3, 22, 10],
      [4, 0, 1],
    ]);

    final testFeatureMatrix = Matrix.fromList([
      [10, 20, 30, 40, 50],
      [11, 22, 33, 44, 55],
      [31, 32, 93, 34, 35],
      [91, 72, 83, 74, 65],
      [55, 11, 99, 33, 12],
      [-11, 0, 123, 1000, 1e-4],
    ]);

    final interceptColumn = Vector.fromList([
      interceptScale,
      interceptScale,
      interceptScale,
      interceptScale,
      interceptScale,
      interceptScale,
    ]);

    final testFeaturesMatrixWithIntercept = Matrix.fromColumns([
      interceptColumn,
      ...testFeatureMatrix.columns,
    ]);

    final mockedProbabilities = Matrix.fromList([
      [0.6, 0.2, 0.2],
      [0.1, 0.8, 0.1],
      [0.1, 0.7, 0.2],
      [0.05, 0.05, 0.9],
      [0.5, 0.4, 0.1],
      [0.05, 0.8, 0.15],
    ]);

    final expectedOutcomeMatrix = Matrix.fromList([
      [positiveLabel, negativeLabel, negativeLabel],
      [negativeLabel, positiveLabel, negativeLabel],
      [negativeLabel, positiveLabel, negativeLabel],
      [negativeLabel, negativeLabel, positiveLabel],
      [positiveLabel, negativeLabel, negativeLabel],
      [negativeLabel, positiveLabel, negativeLabel],
    ]);

    final firstClass = 'target_1';
    final secondClass = 'target_2';
    final thirdClass = 'target_3';
    final targetNames = [firstClass, secondClass, thirdClass];

    final testFeatures = DataFrame.fromMatrix(
      testFeatureMatrix,
      header: ['1', '2', '3', '4', '5', ...targetNames],
    );

    late SoftmaxRegressorImpl regressor;

    setUp(() {
      when(
        linkFunctionMock.link(any),
      ).thenReturn(mockedProbabilities);

      softmaxRegressorInjector
          .registerSingleton<SoftmaxRegressorFactory>(() => classifierFactory);

      regressor = SoftmaxRegressorImpl(
        optimizerType,
        iterationsLimit,
        initialLearningRate,
        decay,
        minCoefficientsUpdate,
        lambda,
        regularizationType,
        randomSeed,
        batchSize,
        isFittingDataNormalized,
        learningRateType,
        initialCoefficientsType,
        initialCoefficients,
        coefficientsByClasses,
        targetNames,
        linkFunctionMock,
        fitIntercept,
        interceptScale,
        positiveLabel,
        negativeLabel,
        costPerIteration,
        dtype,
      );
    });

    tearDown(() {
      reset(linkFunctionMock);
      injector.clearAll();
      softmaxRegressorInjector.clearAll();
    });

    group('default constructor', () {
      test('should throw an exception if no coefficients are provided', () {
        final actual = () => SoftmaxRegressorImpl(
              optimizerType,
              iterationsLimit,
              initialLearningRate,
              decay,
              minCoefficientsUpdate,
              lambda,
              regularizationType,
              randomSeed,
              batchSize,
              isFittingDataNormalized,
              learningRateType,
              initialCoefficientsType,
              initialCoefficients,
              Matrix.empty(),
              targetNames,
              linkFunctionMock,
              fitIntercept,
              interceptScale,
              positiveLabel,
              negativeLabel,
              costPerIteration,
              dtype,
            );

        expect(actual, throwsException);
      });

      test(
          'should throw an exception if coefficients for too few number of '
          'classes are provided', () {
        final actual = () => SoftmaxRegressorImpl(
              optimizerType,
              iterationsLimit,
              initialLearningRate,
              decay,
              minCoefficientsUpdate,
              lambda,
              regularizationType,
              randomSeed,
              batchSize,
              isFittingDataNormalized,
              learningRateType,
              initialCoefficientsType,
              initialCoefficients,
              Matrix.fromList([
                [1],
                [2],
                [3],
              ]),
              targetNames,
              linkFunctionMock,
              fitIntercept,
              interceptScale,
              positiveLabel,
              negativeLabel,
              costPerIteration,
              dtype,
            );

        expect(actual, throwsException);
      });

      test('should persist hyperparameters', () {
        final classifier = SoftmaxRegressorImpl(
          optimizerType,
          iterationsLimit,
          initialLearningRate,
          decay,
          minCoefficientsUpdate,
          lambda,
          regularizationType,
          randomSeed,
          batchSize,
          isFittingDataNormalized,
          learningRateType,
          initialCoefficientsType,
          initialCoefficients,
          coefficientsByClasses,
          targetNames,
          linkFunctionMock,
          fitIntercept,
          interceptScale,
          positiveLabel,
          negativeLabel,
          costPerIteration,
          dtype,
        );

        expect(classifier.optimizerType, optimizerType);
        expect(classifier.iterationsLimit, iterationsLimit);
        expect(classifier.initialLearningRate, initialLearningRate);
        expect(classifier.decay, decay);
        expect(classifier.minCoefficientsUpdate, minCoefficientsUpdate);
        expect(classifier.lambda, lambda);
        expect(classifier.regularizationType, regularizationType);
        expect(classifier.randomSeed, randomSeed);
        expect(classifier.batchSize, batchSize);
        expect(classifier.isFittingDataNormalized, isFittingDataNormalized);
        expect(classifier.learningRateType, learningRateType);
        expect(classifier.initialCoefficientsType, initialCoefficientsType);
        expect(classifier.initialCoefficients, initialCoefficients);
      });
    });

    group('predict', () {
      test('should throw an exception if no features provided', () {
        final testFeatureMatrix = Matrix.empty();
        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);

        expect(() => regressor.predict(testFeatures), throwsException);
      });

      test('should throw an exception if too many features provided', () {
        final testFeatureMatrix = Matrix.fromList([
          [1, 2, 3, 4, 5, 6, 7, 8],
        ]);

        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);

        expect(() => regressor.predict(testFeatures), throwsException);
      });

      test('should throw an exception if too few features provided', () {
        final testFeatureMatrix = Matrix.fromList([
          [1, 2, 3],
        ]);

        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);

        expect(() => regressor.predict(testFeatures), throwsException);
      });

      test('should consider intercept term', () {
        regressor.predict(testFeatures);

        verify(linkFunctionMock.link(
          argThat(
            iterable2dAlmostEqualTo(
                testFeaturesMatrixWithIntercept * coefficientsByClasses),
          ),
        )).called(1);
      });

      test('should use a class with maximal probability for prediction', () {
        final actual = regressor.predict(testFeatures);

        expect(actual.toMatrix(dtype), equals(expectedOutcomeMatrix));
      });

      test('should predict the first class if outcome is equiprobable', () {
        reset(linkFunctionMock);

        when(
          linkFunctionMock.link(any),
        ).thenReturn(
          Matrix.fromList([
            [0.33, 0.33, 0.33],
          ]),
        );

        final actual = regressor.predict(testFeatures);
        final expectedOutcome = Matrix.fromList([
          [positiveLabel, negativeLabel, negativeLabel],
        ]);

        expect(actual.toMatrix(dtype), equals(expectedOutcome));
      });

      test('should return a dataframe with a proper header', () {
        final actual = regressor.predict(testFeatures);

        expect(actual.header, equals(targetNames));
      });
    });

    group('predictProbabilities', () {
      test('should throw an exception if no features provided', () {
        final testFeatures = DataFrame.fromMatrix(Matrix.empty());

        expect(() => regressor.predictProbabilities(testFeatures),
            throwsException);
      });

      test('should throw an exception if too few features provided', () {
        final testFeatures = DataFrame.fromMatrix(Matrix.fromList([
          [1, 2],
        ]));

        expect(() => regressor.predictProbabilities(testFeatures),
            throwsException);
      });

      test('should throw an exception if too many features provided', () {
        final testFeatures = DataFrame.fromMatrix(Matrix.fromList([
          [1, 2, 4, 4, 5, 6],
        ]));

        expect(() => regressor.predictProbabilities(testFeatures),
            throwsException);
      });

      test('should consider intercept term', () {
        regressor.predictProbabilities(testFeatures);

        verify(linkFunctionMock
                .link(testFeaturesMatrixWithIntercept * coefficientsByClasses))
            .called(1);
      });

      test('should return probabilities as dataframe', () {
        final probabilities = regressor.predictProbabilities(testFeatures);

        expect(probabilities.rows, equals(mockedProbabilities));
      });

      test('should return a dataframe with a proper header', () {
        final probabilities = regressor.predictProbabilities(testFeatures);

        expect(probabilities.header, equals(targetNames));
      });

      test('should persist cost per iteration list', () {
        expect(regressor.costPerIteration, costPerIteration);
      });
    });

    group('retrain', () {
      test('should call a factory while retraining the model', () {
        regressor.retrain(retrainingData);

        verify(classifierFactory.create(
          trainData: retrainingData,
          targetNames: targetNames,
          optimizerType: optimizerType,
          iterationsLimit: iterationsLimit,
          initialLearningRate: initialLearningRate,
          decay: decay,
          minCoefficientsUpdate: minCoefficientsUpdate,
          lambda: lambda,
          regularizationType: regularizationType,
          randomSeed: randomSeed,
          batchSize: batchSize,
          fitIntercept: fitIntercept,
          interceptScale: interceptScale,
          learningRateType: learningRateType,
          isFittingDataNormalized: isFittingDataNormalized,
          initialCoefficientsType: initialCoefficientsType,
          initialCoefficients: initialCoefficients,
          positiveLabel: positiveLabel,
          negativeLabel: negativeLabel,
          dtype: dtype,
          collectLearningData: collectLearningData,
        )).called(1);
      });

      test('should return a new instance as a retrained model', () {
        final retrainedModel = regressor.retrain(retrainingData);

        expect(retrainedModel, same(retrainedModelMock));
        expect(retrainedModel, isNot(same(regressor)));
      });

      test('should have a proper jsdon schema version', () {
        final model = SoftmaxRegressorImpl(
          optimizerType,
          iterationsLimit,
          initialLearningRate,
          decay,
          minCoefficientsUpdate,
          lambda,
          regularizationType,
          randomSeed,
          batchSize,
          isFittingDataNormalized,
          learningRateType,
          initialCoefficientsType,
          initialCoefficients,
          coefficientsByClasses,
          targetNames,
          linkFunctionMock,
          fitIntercept,
          interceptScale,
          positiveLabel,
          negativeLabel,
          costPerIteration,
          dtype,
        );

        expect(model.schemaVersion, 4);
      });
    });
  });
}
