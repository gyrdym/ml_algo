import 'package:ml_algo/src/classifier/logistic_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_factory.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_impl.dart';
import 'package:ml_algo/src/common/exception/invalid_class_labels_exception.dart';
import 'package:ml_algo/src/common/exception/invalid_probability_threshold_exception.dart';
import 'package:ml_algo/src/common/exception/outdated_json_schema_exception.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../../mocks.dart';

void main() {
  group('LogisticRegressorImpl', () {
    final defaultOptimizerType = LinearOptimizerType.gradient;
    final defaultIterationsLimit = 123;
    final defaultInitialLearningRate = 0.75;
    final defaultMinCoefficientsUpdate = 0.5;
    final defaultLambda = 11.0;
    final defaultRegularizationType = RegularizationType.L2;
    final defaultRandomSeed = 1234;
    final defaultBatchSize = 100;
    final defaultIsFittingDataNormalizedFlag = false;
    final defaultLearningRateType = LearningRateType.decreasingAdaptive;
    final defaultInitialCoefficientsType = InitialCoefficientsType.zeroes;
    final defaultInitialCoefficients = Vector.fromList([1, 2, 3, 4, 5]);
    final linkFunctionMock = LinkFunctionMock();
    final className = 'class 1';
    final defaultFitIntercept = true;
    final defaultInterceptScale = 10.0;
    final defaultCoefficients = Matrix.column([1, 2, 3, 4]);
    final defaultProbabilityThreshold = 0.6;
    final defaultNegativeLabel = -1;
    final defaultPositiveLabel = 1;
    final defaultDType = DType.float32;
    final defaultCostPerIteration = [1.2, 233.01, 23, -10001];
    final retrainingData = DataFrame([[10, -10, 20, -20]]);
    final retrainedModelMock = LogisticRegressorMock();
    final classifierFactory = createLogisticRegressorFactoryMock(
        retrainedModelMock);
    final createRegressor = ({
      LinearOptimizerType optimizerType,
      int iterationsLimit,
      double initialLearningRate,
      double minCoefficientsUpdate,
      double lambda,
      RegularizationType regularizationType,
      int randomSeed,
      int batchSize,
      bool isFittingDataNormalized,
      LearningRateType learningRateType,
      InitialCoefficientsType initialCoefficientsType,
      Vector initialCoefficients,
      Iterable<String> targetNames,
      LinkFunction linkFunction,
      bool fitIntercept,
      double interceptScale,
      Matrix coefficients,
      double probabilityThreshold,
      num negativeLabel,
      num positiveLabel,
      List<num> costPerIteration,
      DType dtype,
    }) => LogisticRegressorImpl(
      optimizerType ?? defaultOptimizerType,
      iterationsLimit ?? defaultIterationsLimit,
      initialLearningRate ?? defaultInitialLearningRate,
      minCoefficientsUpdate ?? defaultMinCoefficientsUpdate,
      lambda ?? defaultLambda,
      regularizationType ?? defaultRegularizationType,
      randomSeed ?? defaultRandomSeed,
      batchSize ?? defaultBatchSize,
      isFittingDataNormalized ?? defaultIsFittingDataNormalizedFlag,
      learningRateType ?? defaultLearningRateType,
      initialCoefficientsType ?? defaultInitialCoefficientsType,
      initialCoefficients ?? defaultInitialCoefficients,
      targetNames ?? [className],
      linkFunction ?? linkFunctionMock,
      fitIntercept ?? defaultFitIntercept,
      interceptScale ?? defaultInterceptScale,
      coefficients ?? defaultCoefficients,
      probabilityThreshold ?? defaultProbabilityThreshold,
      negativeLabel ?? defaultNegativeLabel,
      positiveLabel ?? defaultPositiveLabel,
      costPerIteration ?? defaultCostPerIteration,
      dtype ?? defaultDType,
    );

    final testFeatureMatrix = Matrix.fromList([
      [20, 30, 40],
      [20, 22, 11],
      [90, 87, 52],
      [12, 20, 21],
      [33, 44, 55],
    ]);

    final featuresWithIntercept = Matrix.fromList([
      [defaultInterceptScale, 20, 30, 40],
      [defaultInterceptScale, 20, 22, 11],
      [defaultInterceptScale, 90, 87, 52],
      [defaultInterceptScale, 12, 20, 21],
      [defaultInterceptScale, 33, 44, 55],
    ]);

    final mockedProbabilities = Matrix.column([0.8, 0.5, 0.6, 0.7, 0.3]);

    setUp(() {
      logisticRegressorInjector
          .registerSingleton<LogisticRegressorFactory>(() => classifierFactory);

      when(linkFunctionMock.link(any)).thenReturn(mockedProbabilities);
    });

    tearDown(() {
      reset(linkFunctionMock);
      injector.clearAll();
      logisticRegressorInjector.clearAll();
    });

    group('default constructor', () {
      test('should create the instance with `classNames` list of just one '
          'element', () {
        expect(createRegressor().targetNames, equals([className]));
      });

      test('should throw an exception if probability threshold is less '
          'than 0', () {
        final probabilityThreshold = -0.1;
        final actual = () => createRegressor(
            probabilityThreshold: probabilityThreshold,
        );

        expect(actual, throwsA(isA<InvalidProbabilityThresholdException>()));
      });

      test('should throw an exception if probability threshold is less '
          'equal to 0', () {
        final probabilityThreshold = 0.0;
        final actual = () => createRegressor(
          probabilityThreshold: probabilityThreshold,
        );

        expect(actual, throwsA(isA<InvalidProbabilityThresholdException>()));
      });

      test('should throw an exception if probability threshold is equal '
          'to 1', () {
        final probabilityThreshold = 1.0;
        final actual = () => createRegressor(
          probabilityThreshold: probabilityThreshold,
        );

        expect(actual, throwsA(isA<InvalidProbabilityThresholdException>()));
      });

      test('should throw an exception if probability threshold is greater '
          'than 1', () {
        final probabilityThreshold = 1.2;
        final actual = () => createRegressor(
          probabilityThreshold: probabilityThreshold,
        );

        expect(actual, throwsA(isA<InvalidProbabilityThresholdException>()));
      });

      test('should throw an exception if positive class label equals to '
          'negative class label', () {
        final negativeLabel = 1000;
        final positiveLabel = 1000;
        final actual = () => createRegressor(
          negativeLabel: negativeLabel,
          positiveLabel: positiveLabel,
        );

        expect(actual, throwsA(isA<InvalidClassLabelsException>()));
      });

      test('should throw an exception if no coefficients are provided', () {
        final coefficients = Matrix.empty();
        final actual = () => createRegressor(
          coefficients: coefficients,
        );

        expect(actual, throwsException);
      });

      test('should throw an exception if coefficients provided for more than '
          'one class', () {
        final coefficients = Matrix.fromList([
          [1, 2, 3],
          [7, 5, 8],
          [2, 0, 9],
          [1, 3, 3],
        ]);
        final actual = () => createRegressor(
          coefficients: coefficients,
        );

        expect(actual, throwsException);
      });

      test('should persist cost per iteration', () {
        expect(createRegressor().costPerIteration, defaultCostPerIteration);
      });

      test('should persist hyperparameters', () {
        final model = createRegressor();

        expect(model.optimizerType, defaultOptimizerType);
        expect(model.iterationsLimit, defaultIterationsLimit);
        expect(model.initialLearningRate, defaultInitialLearningRate);
        expect(model.minCoefficientsUpdate, defaultMinCoefficientsUpdate);
        expect(model.probabilityThreshold, defaultProbabilityThreshold);
        expect(model.lambda, defaultLambda);
        expect(model.regularizationType, defaultRegularizationType);
        expect(model.randomSeed, defaultRandomSeed);
        expect(model.batchSize, defaultBatchSize);
        expect(model.isFittingDataNormalized, defaultIsFittingDataNormalizedFlag);
        expect(model.learningRateType, defaultLearningRateType);
        expect(model.initialCoefficientsType, defaultInitialCoefficientsType);
        expect(model.initialCoefficients, defaultInitialCoefficients);
      });
    });

    group('predict', () {
      test('should throw an exception if no test features are provided', () {
        final testFeatures = DataFrame.fromMatrix(Matrix.empty());

        expect(() => createRegressor().predict(testFeatures), throwsException);
      });

      test('should consider intercept term', () {
        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);

        createRegressor().predict(testFeatures);

        verify(linkFunctionMock.link(featuresWithIntercept * defaultCoefficients))
            .called(1);
      });

      test('should throw an error if too many features are provided', () {
        final testFeatureMatrix = Matrix.fromList([
          [10, 20, 30, 40],
        ]);
        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);

        expect(() => createRegressor().predict(testFeatures), throwsException);
      });

      test('should throw an error if too few features are provided', () {
        final testFeatureMatrix = Matrix.fromList([
          [10, 20],
        ]);
        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);

        expect(() => createRegressor().predict(testFeatures), throwsException);
      });

      test('should predict class labels basing on calculated probabilities', () {
        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);
        final prediction = createRegressor().predict(testFeatures);

        expect(prediction.rows, equals([
          [defaultPositiveLabel],
          [defaultNegativeLabel],
          [defaultPositiveLabel],
          [defaultPositiveLabel],
          [defaultNegativeLabel],
        ]));
      });

      test('should return a dataframe with a proper header', () {
        final testFeatures = DataFrame.fromMatrix(testFeatureMatrix);
        final prediction = createRegressor().predict(testFeatures);

        expect(prediction.header, equals([className]));
      });
    });

    group('retrain', () {
      test('should call a factory while retraining the model', () {
        createRegressor().retrain(retrainingData);

        verify(classifierFactory.create(
          trainData: retrainingData,
          targetName: className,
          optimizerType: defaultOptimizerType,
          iterationsLimit: defaultIterationsLimit,
          initialLearningRate: defaultInitialLearningRate,
          minCoefficientsUpdate: defaultMinCoefficientsUpdate,
          lambda: defaultLambda,
          regularizationType: defaultRegularizationType,
          randomSeed: defaultRandomSeed,
          batchSize: defaultBatchSize,
          isFittingDataNormalized: defaultIsFittingDataNormalizedFlag,
          learningRateType: defaultLearningRateType,
          initialCoefficientsType: defaultInitialCoefficientsType,
          initialCoefficients: defaultInitialCoefficients,
          fitIntercept: defaultFitIntercept,
          interceptScale: defaultInterceptScale,
          probabilityThreshold: defaultProbabilityThreshold,
          negativeLabel: defaultNegativeLabel,
          positiveLabel: defaultPositiveLabel,
          collectLearningData: false,
          dtype: defaultDType,
        )).called(1);
      });

      test('should return a new instance as a retrained model', () {
        final regressor = createRegressor();
        final retrainedModel = regressor.retrain(retrainingData);

        expect(retrainedModel, same(retrainedModelMock));
        expect(retrainedModel, isNot(same(regressor)));
      });

      test('should throw exception if the model schema is outdated or '
          'null', () {
        final model = LogisticRegressorImpl(
          defaultOptimizerType,
          defaultIterationsLimit,
          defaultInitialLearningRate,
          defaultMinCoefficientsUpdate,
          defaultLambda,
          defaultRegularizationType,
          defaultRandomSeed,
          defaultBatchSize,
          defaultIsFittingDataNormalizedFlag,
          defaultLearningRateType,
          defaultInitialCoefficientsType,
          defaultInitialCoefficients,
          [className],
          linkFunctionMock,
          defaultFitIntercept,
          defaultInterceptScale,
          defaultCoefficients,
          defaultProbabilityThreshold,
          defaultNegativeLabel,
          defaultPositiveLabel,
          defaultCostPerIteration,
          defaultDType,
          schemaVersion: null,
        );

        expect(() => model.retrain(retrainingData),
            throwsA(isA<OutdatedJsonSchemaException>()));
      });

      test('should have a proper json schema version', () {
        expect(createRegressor().schemaVersion, 2);
      });
    });
  });
}
