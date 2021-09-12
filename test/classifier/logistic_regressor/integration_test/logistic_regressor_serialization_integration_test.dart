import 'dart:io';

import 'package:ml_algo/src/classifier/logistic_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_constants.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_json_keys.dart';
import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type_json_encoded_values.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type_json_encoded_values.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type_json_encoded_values.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type_json_encoded_values.dart';
import 'package:ml_algo/src/link_function/inverse_logit_link_function.dart';
import 'package:ml_algo/src/link_function/link_function_encoded_values.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/dtype_to_json.dart';
import 'package:ml_linalg/matrix_to_json.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() {
  final data = <Iterable<num>>[
    [5.0, 7.0, 6.0, 1.0],
    [1.0, 2.0, 3.0, 0.0],
    [10.0, 12.0, 31.0, 0.0],
    [9.0, 8.0, 5.0, 0.0],
    [4.0, 0.0, 1.0, 1.0],
  ];
  final targetName = 'col_3';
  final samples = DataFrame(data, headerExists: false);
  final fileName = 'test/classifier/logistic_regressor/logistic_regressor.json';

  final interceptScale1 = 10.0;
  final interceptScale2 = -100.0;
  final interceptScale3 = 0.0;

  final dtype1 = DType.float32;
  final dtype2 = DType.float64;

  final probabilityThreshold1 = 0.1;
  final probabilityThreshold2 = 0.9;

  final positiveLabel1 = 100;
  final positiveLabel2 = -100;
  final positiveLabel3 = 10;

  final negativeLabel1 = 101;
  final negativeLabel2 = -101;
  final negativeLabel3 = 11;

  final createClassifier = ({
    LinearOptimizerType optimizerType = LinearOptimizerType.gradient,
    int iterationsLimit = 2,
    double minCoefficientsUpdate = 1e-12,
    double initialLearningRate = 1.0,
    double lambda = 0.0,
    RegularizationType? regularizationType,
    int? randomSeed,
    int batchSize = 5,
    bool isFittingDataNormalized = false,
    LearningRateType learningRateType = LearningRateType.constant,
    InitialCoefficientsType initialCoefficientsType =
        InitialCoefficientsType.zeroes,
    Vector? initialCoefficients,
    String targetName = 'col_3',
    bool fitIntercept = false,
    double interceptScale = 3.0,
    double probabilityThreshold = 0.9,
    num positiveLabel = 1,
    num negativeLabel = 0,
    DType dtype = DType.float32,
    bool collectLearningData = false,
  }) =>
      LogisticRegressor(
        samples,
        targetName,
        iterationsLimit: iterationsLimit,
        minCoefficientsUpdate: minCoefficientsUpdate,
        optimizerType: optimizerType,
        initialLearningRate: initialLearningRate,
        lambda: lambda,
        regularizationType: regularizationType,
        batchSize: batchSize,
        randomSeed: randomSeed,
        isFittingDataNormalized: isFittingDataNormalized,
        learningRateType: learningRateType,
        initialCoefficientsType: initialCoefficientsType,
        initialCoefficients: initialCoefficients,
        fitIntercept: fitIntercept,
        interceptScale: interceptScale,
        dtype: dtype,
        probabilityThreshold: probabilityThreshold,
        positiveLabel: positiveLabel,
        negativeLabel: negativeLabel,
        collectLearningData: collectLearningData,
      );

  group('LogistiRegressor.toJson', () {
    test('should serialize coefficientsByClasses field', () {
      final classifier = createClassifier();
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorCoefficientsByClassesJsonKey],
        matrixToJson(classifier.coefficientsByClasses),
      );
    });

    test('should serialize optimizerType field, gradient_optimizer optimizer',
        () {
      final classifier = createClassifier(
        optimizerType: LinearOptimizerType.gradient,
      );
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorLinearOptimizerTypeJsonKey],
        gradientLinearOptimizerTypeEncodedValue,
      );
    });

    // coordinate optimization is not implemented yet for LogisticRegressor
    test('should serialize optimizerType field, coordinate optimizer', () {
      final classifier = createClassifier(
        optimizerType: LinearOptimizerType.coordinate,
      );
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorLinearOptimizerTypeJsonKey],
        coordinateLinearOptimizerTypeEncodedValue,
      );
    }, skip: true);

    test('should serialize iterationsLimit field, iterationsLimit=100', () {
      final classifier = createClassifier(
        iterationsLimit: 100,
      );
      final serialized = classifier.toJson();

      expect(serialized[logisticRegressorIterationsLimitJsonKey], 100);
    });

    test('should serialize initialLearningRate field, initialLearningRate=10',
        () {
      final classifier = createClassifier(
        initialLearningRate: 10,
      );
      final serialized = classifier.toJson();

      expect(serialized[logisticRegressorInitialLearningRateJsonKey], 10);
    });

    test(
        'should serialize minCoefficientsUpdate field, '
        'minCoefficientsUpdate=100', () {
      final classifier = createClassifier(
        minCoefficientsUpdate: 100,
      );
      final serialized = classifier.toJson();

      expect(serialized[logisticRegressorMinCoefsUpdateJsonKey], 100);
    });

    test('should serialize lambda field', () {
      final classifier = createClassifier(
        lambda: 199.0,
      );
      final serialized = classifier.toJson();

      expect(serialized[logisticRegressorLambdaJsonKey], 199.0);
    });

    // coordinate optimization is not implemented yet for LogisticRegressor
    test('should serialize regularizationType field, regularizationType=L1',
        () {
      final classifier = createClassifier(
        optimizerType: LinearOptimizerType.coordinate,
        regularizationType: RegularizationType.L1,
      );
      final serialized = classifier.toJson();

      expect(serialized[logisticRegressorRegularizationTypeJsonKey],
          l1RegularizationTypeJsonEncodedValue);
    }, skip: true);

    test('should serialize regularizationType field, regularizationType=L2',
        () {
      final classifier = createClassifier(
        optimizerType: LinearOptimizerType.gradient,
        regularizationType: RegularizationType.L2,
      );
      final serialized = classifier.toJson();

      expect(serialized[logisticRegressorRegularizationTypeJsonKey],
          l2RegularizationTypeJsonEncodedValue);
    });

    test('should serialize regularizationType field, regularizationType=null',
        () {
      final classifier = createClassifier(
        optimizerType: LinearOptimizerType.gradient,
        regularizationType: null,
      );
      final serialized = classifier.toJson();

      expect(serialized.containsKey(logisticRegressorRegularizationTypeJsonKey),
          false);
    });

    test('should serialize randomSeed field, randomSeed=100', () {
      final classifier = createClassifier(
        randomSeed: 100,
      );
      final serialized = classifier.toJson();

      expect(serialized[logisticRegressorRandomSeedJsonKey], 100);
    });

    test('should serialize randomSeed field, randomSeed=null', () {
      final classifier = createClassifier(
        randomSeed: null,
      );
      final serialized = classifier.toJson();

      expect(serialized.containsKey(logisticRegressorRandomSeedJsonKey), false);
    });

    test('should serialize batchSize field', () {
      final classifier = createClassifier(
        batchSize: 4,
      );
      final serialized = classifier.toJson();

      expect(serialized[logisticRegressorBatchSizeJsonKey], 4);
    });

    test(
        'should serialize isFittingDataNormalized field, '
        'isFittingDataNormalized=true', () {
      final classifier = createClassifier(
        isFittingDataNormalized: true,
      );
      final serialized = classifier.toJson();

      expect(serialized[logisticRegressorDataNormalizedFlagJsonKey], true);
    });

    test(
        'should serialize isFittingDataNormalized field, '
        'isFittingDataNormalized=false', () {
      final classifier = createClassifier(
        isFittingDataNormalized: false,
      );
      final serialized = classifier.toJson();

      expect(serialized[logisticRegressorDataNormalizedFlagJsonKey], false);
    });

    test(
        'should serialize learningRateType field, '
        'learningRateType=timeBased', () {
      final classifier = createClassifier(
        learningRateType: LearningRateType.timeBased,
      );
      final serialized = classifier.toJson();

      expect(serialized[logisticRegressorLearningRateTypeJsonKey],
          learningRateTypeToEncodedValue[LearningRateType.timeBased]);
    });

    test(
        'should serialize learningRateType field, '
        'learningRateType=constant', () {
      final classifier = createClassifier(
        learningRateType: LearningRateType.constant,
      );
      final serialized = classifier.toJson();

      expect(serialized[logisticRegressorLearningRateTypeJsonKey],
          learningRateTypeToEncodedValue[LearningRateType.constant]);
    });

    test(
        'should serialize initialCoefficientsType field, '
        'initialCoefficientsType=zero', () {
      final classifier = createClassifier(
        initialCoefficientsType: InitialCoefficientsType.zeroes,
      );
      final serialized = classifier.toJson();

      expect(serialized[logisticRegressorInitCoefficientsTypeJsonKey],
          zeroesInitialCoefficientsTypeJsonEncodedValue);
    });

    test(
        'should serialize initialCoefficients field, '
        'initialCoefficients=[2, 2, 2]', () {
      final classifier = createClassifier(
        initialCoefficients: Vector.fromList([2, 2, 2]),
      );
      final serialized = classifier.toJson();

      expect(serialized[logisticRegressorInitCoefficientsJsonKey],
          Vector.fromList([2, 2, 2]).toJson());
    });

    test('should serialize classNames field', () {
      final classifier = createClassifier();
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorClassNamesJsonKey],
        [targetName],
      );
    });

    test('should serialize fitIntercept field, fitIntercept=true', () {
      final fitIntercept = true;
      final classifier = createClassifier(fitIntercept: fitIntercept);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorFitInterceptJsonKey],
        fitIntercept,
      );
    });

    test('should serialize fitIntercept field, fitIntercept=false', () {
      final fitIntercept = false;
      final classifier = createClassifier(fitIntercept: fitIntercept);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorFitInterceptJsonKey],
        fitIntercept,
      );
    });

    test(
        'should serialize interceptScale field, '
        'interceptScale=$interceptScale1', () {
      final classifier = createClassifier(interceptScale: interceptScale1);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorInterceptScaleJsonKey],
        interceptScale1,
      );
    });

    test(
        'should serialize interceptScale field, '
        'interceptScale=$interceptScale2', () {
      final classifier = createClassifier(interceptScale: interceptScale2);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorInterceptScaleJsonKey],
        interceptScale2,
      );
    });

    test(
        'should serialize interceptScale field, '
        'interceptScale=$interceptScale3', () {
      final classifier = createClassifier(interceptScale: interceptScale3);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorInterceptScaleJsonKey],
        interceptScale3,
      );
    });

    test('should serialize dtype field, dtype=$dtype1', () {
      final classifier = createClassifier(dtype: dtype1);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorDTypeJsonKey],
        dTypeToJson(dtype1),
      );
    });

    test('should serialize dtype field, dtype=$dtype2', () {
      final classifier = createClassifier(dtype: dtype2);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorDTypeJsonKey],
        dTypeToJson(dtype2),
      );
    });

    test(
        'should serialize probabilityThreshold field, '
        'value=$probabilityThreshold1', () {
      final classifier =
          createClassifier(probabilityThreshold: probabilityThreshold1);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorProbabilityThresholdJsonKey],
        probabilityThreshold1,
      );
    });

    test(
        'should serialize probabilityThreshold field, '
        'value=$probabilityThreshold2', () {
      final classifier =
          createClassifier(probabilityThreshold: probabilityThreshold2);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorProbabilityThresholdJsonKey],
        probabilityThreshold2,
      );
    });

    test('should serialize positiveLabel field, value=$positiveLabel1', () {
      final classifier = createClassifier(positiveLabel: positiveLabel1);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorPositiveLabelJsonKey],
        positiveLabel1,
      );
    });

    test('should serialize positiveLabel field, value=$positiveLabel2', () {
      final classifier = createClassifier(positiveLabel: positiveLabel2);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorPositiveLabelJsonKey],
        positiveLabel2,
      );
    });

    test('should serialize positiveLabel field, value=$positiveLabel3', () {
      final classifier = createClassifier(positiveLabel: positiveLabel3);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorPositiveLabelJsonKey],
        positiveLabel3,
      );
    });

    test('should serialize negativeLabel field, value=$negativeLabel1', () {
      final classifier = createClassifier(negativeLabel: negativeLabel1);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorNegativeLabelJsonKey],
        negativeLabel1,
      );
    });

    test('should serialize negativeLabel field, value=$negativeLabel2', () {
      final classifier = createClassifier(negativeLabel: negativeLabel2);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorNegativeLabelJsonKey],
        negativeLabel2,
      );
    });

    test('should serialize negativeLabel field, value=$negativeLabel3', () {
      final classifier = createClassifier(negativeLabel: negativeLabel3);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorNegativeLabelJsonKey],
        negativeLabel3,
      );
    });

    test('should serialize linkFunction field', () {
      final classifier = createClassifier();
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorLinkFunctionJsonKey],
        inverseLogitLinkFunctionEncoded,
      );
    });

    test('should serialize cost per iteration list', () {
      final classifier = createClassifier(collectLearningData: true);
      final serialized = classifier.toJson();

      expect(serialized[logisticRegressorCostPerIterationJsonKey],
          classifier.costPerIteration);
    });

    test('should serialize schemaVersion field', () {
      final classifier = createClassifier(collectLearningData: true);
      final serialized = classifier.toJson();

      expect(serialized[jsonSchemaVersionJsonKey],
          logisticRegressorJsonSchemaVersion);
    });
  });

  group('LogisticRegressor.saveAsJson', () {
    tearDown(() async {
      final file = File(fileName);

      if (await file.exists()) {
        await file.delete();
      }

      injector.clearAll();
      logisticRegressorInjector.clearAll();
    });

    test(
        'should return a pointer to a json file while saving serialized '
        'data', () async {
      final classifier = createClassifier();
      final file = await classifier.saveAsJson(fileName);

      expect(await file.exists(), isTrue);
      expect(file.path, fileName);
    });

    test(
        'should restore a classifier instance from json file, '
        'dtype=DType.float32', () async {
      final classifier = createClassifier(dtype: DType.float32);
      await classifier.saveAsJson(fileName);

      final file = File(fileName);
      final encodedData = await file.readAsString();
      final restoredClassifier = LogisticRegressor.fromJson(encodedData);

      expect(restoredClassifier.coefficientsByClasses,
          classifier.coefficientsByClasses);
      expect(restoredClassifier.interceptScale, classifier.interceptScale);
      expect(restoredClassifier.fitIntercept, classifier.fitIntercept);
      expect(restoredClassifier.dtype, classifier.dtype);
      expect(restoredClassifier.linkFunction, isA<InverseLogitLinkFunction>());
      expect(restoredClassifier.targetNames, [targetName]);
    });

    test(
        'should restore a classifier instance from json file, '
        'dtype=DType.float64', () async {
      final classifier = createClassifier(dtype: DType.float64);
      await classifier.saveAsJson(fileName);

      final file = File(fileName);
      final encodedData = await file.readAsString();
      final restoredClassifier = LogisticRegressor.fromJson(encodedData);

      expect(restoredClassifier.coefficientsByClasses,
          classifier.coefficientsByClasses);
      expect(restoredClassifier.interceptScale, classifier.interceptScale);
      expect(restoredClassifier.fitIntercept, classifier.fitIntercept);
      expect(restoredClassifier.dtype, classifier.dtype);
      expect(restoredClassifier.linkFunction, isA<InverseLogitLinkFunction>());
      expect(restoredClassifier.targetNames, [targetName]);
    });
  });
}
