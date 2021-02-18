import 'dart:io';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_constants.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_json_keys.dart';
import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type_json_encoded_values.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type_json_encoded_values.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type_json_encoded_values.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type_json_encoded_values.dart';
import 'package:ml_algo/src/link_function/link_function_encoded_values.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

void main() {
  final featureNames = ['feature 1', 'feature 2', 'feature 3'];
  final targetNames = ['class_1', 'class_2', 'class_3'];
  final fileName = 'test/classifier/softmax_regressor/softmax_regressor.json';

  final interceptScale1 = -100.0;
  final interceptScale2 = 0.0;
  final interceptScale3 = 12.0;

  final positiveLabel1 = -100;
  final positiveLabel2 = 0;
  final positiveLabel3 = 140;

  final negativeLabel1 = -1;
  final negativeLabel2 = 2000;
  final negativeLabel3 = 0;

  final initialCoeffs = Matrix.fromList([
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3],
  ]);

  final createClassifier = ({
    LinearOptimizerType optimizerType = LinearOptimizerType.gradient,
    int iterationsLimit = 100,
    double initialLearningRate = 1e-3,
    double minCoefficientsUpdate = 1e-12,
    double lambda,
    RegularizationType regularizationType,
    int randomSeed,
    int batchSize = 1,
    bool fitIntercept = true,
    double interceptScale = 10.0,
    LearningRateType learningRateType = LearningRateType.constant,
    bool isFittingDataNormalized,
    InitialCoefficientsType initialCoefficientsType =
        InitialCoefficientsType.zeroes,
    Matrix initialCoefficients,
    num positiveLabel = 1,
    num negativeLabel = -1,
    bool collectLearningData = false,
    DType dtype = DType.float32,
  }) {
    final sourceData = <Iterable<dynamic>>[
      <String>[...featureNames, ...targetNames],
      <num>[   100,    200, 300.89, positiveLabel, negativeLabel, negativeLabel],
      <num>[   444,   20.7, 300.89, negativeLabel, negativeLabel, positiveLabel],
      <num>[   100, -20000, -0.003, negativeLabel, negativeLabel, positiveLabel],
      <num>[   100,    200,    1e5, negativeLabel, positiveLabel, negativeLabel],
      <num>[-0.874, 932.12,   0.98, positiveLabel, negativeLabel, negativeLabel],
    ];
    final dataFrame = DataFrame(sourceData, headerExists: true);

    return SoftmaxRegressor(
      dataFrame,
      targetNames,
      optimizerType: optimizerType,
      iterationsLimit: iterationsLimit,
      initialLearningRate: initialLearningRate,
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
      collectLearningData: collectLearningData,
      dtype: dtype,
    );
  };

  group('SoftmaxRegressor.toJson', () {
    tearDown(() async {
      await module.reset();
      await softmaxRegressorModule.reset();
    });

    test('should serialize optimizerType field', () {
      final classifier = createClassifier();
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorOptimizerTypeJsonKey],
          gradientLinearOptimizerTypeEncodedValue);
    });

    test('should serialize iterationsLimit field', () {
      final classifier = createClassifier(iterationsLimit: 3);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorIterationsLimitJsonKey], 3);
    });

    test('should serialize iterationsLimit field, iterationsLimit=null', () {
      final classifier = createClassifier(
        iterationsLimit: null,
        minCoefficientsUpdate: 1,
      );
      final serialized = classifier.toJson();

      expect(serialized
          .containsKey(softmaxRegressorIterationsLimitJsonKey), false);
    });

    test('should serialize initialLearningRate field', () {
      final classifier = createClassifier(initialLearningRate: 5.5);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorInitialLearningRateJsonKey], 5.5);
    });

    test('should serialize minCoefficientsUpdate field', () {
      final classifier = createClassifier(minCoefficientsUpdate: 7);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorMinCoefsUpdateJsonKey], 7);
    });

    test('should serialize minCoefficientsUpdate field, '
        'minCoefficientsUpdate=null', () {

      final classifier = createClassifier(
        iterationsLimit: 7,
        minCoefficientsUpdate: null,
      );
      final serialized = classifier.toJson();

      expect(serialized
          .containsKey(softmaxRegressorMinCoefsUpdateJsonKey), false);
    });

    test('should serialize lambda field', () {
      final classifier = createClassifier(lambda: 341);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorLambdaJsonKey], 341);
    });

    test('should serialize lambda field, lambda=null', () {
      final classifier = createClassifier(lambda: null);
      final serialized = classifier.toJson();

      expect(serialized.containsKey(softmaxRegressorLambdaJsonKey), false);
    });

    test('should serialize regularizationType field', () {
      final classifier = createClassifier(
          regularizationType: RegularizationType.L2);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorRegularizationTypeJsonKey],
          l2RegularizationTypeJsonEncodedValue);
    });

    test('should serialize regularizationType field, '
        'regularizationType=null', () {

      final classifier = createClassifier(
          regularizationType: null);
      final serialized = classifier.toJson();

      expect(serialized
          .containsKey(softmaxRegressorRegularizationTypeJsonKey), false);
    });

    test('should serialize randomSeed field', () {
      final classifier = createClassifier(randomSeed: 100009);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorRandomSeedJsonKey], 100009);
    });

    test('should serialize randomSeed field, randomSeed=null', () {
      final classifier = createClassifier(randomSeed: null);
      final serialized = classifier.toJson();

      expect(serialized.containsKey(softmaxRegressorRandomSeedJsonKey), false);
    });

    test('should serialize batchSize field', () {
      final classifier = createClassifier(batchSize: 2);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorBatchSizeJsonKey], 2);
    });

    test('should serialize isFittingDataNormalized field, '
        'isFittingDataNormalized=true', () {

      final classifier = createClassifier(isFittingDataNormalized: true);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorFittingDataNormalizedFlagJsonKey],
          true);
    });

    test('should serialize isFittingDataNormalized field, '
        'isFittingDataNormalized=false', () {

      final classifier = createClassifier(isFittingDataNormalized: false);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorFittingDataNormalizedFlagJsonKey],
          false);
    });

    test('should serialize learningRateType field, '
        'decreasingAdaptive type', () {
      final classifier = createClassifier(
          learningRateType: LearningRateType.decreasingAdaptive);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorLearningRateTypeJsonKey],
          decreasingAdaptiveLearningRateTypeJsonEncodedValue);
    });

    test('should serialize learningRateType field, '
        'constant type', () {
      final classifier = createClassifier(
          learningRateType: LearningRateType.constant);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorLearningRateTypeJsonKey],
          constantLearningRateTypeJsonEncodedValue);
    });

    test('should serialize initialCoefficientsType field, '
        'initialCoefficientsType=InitialCoefficientsType.zeros', () {
      final classifier = createClassifier(
          initialCoefficientsType: InitialCoefficientsType.zeroes);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorInitialCoefsTypeJsonKey],
          zeroesInitialCoefficientsTypeJsonEncodedValue);
    });

    test('should serialize initialCoefficients field', () {
      final classifier = createClassifier(
          initialCoefficients: initialCoeffs);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorInitialCoefsJsonKey], initialCoeffs.toJson());
    });

    test('should serialize initialCoefficients field, '
        'initialCoefficients=null', () {
      final classifier = createClassifier(
          initialCoefficients: null);
      final serialized = classifier.toJson();

      expect(serialized.containsKey(softmaxRegressorInitialCoefsJsonKey),
          false);
    });

    test('should serialize classNames field', () {
      final classifier = createClassifier();
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorClassNamesJsonKey], targetNames);
    });

    test('should serialize fitIntercept field, fitIntercept=true', () {
      final classifier = createClassifier(fitIntercept: true);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorFitInterceptJsonKey], true);
    });

    test('should serialize fitIntercept field, fitIntercept=false', () {
      final classifier = createClassifier(fitIntercept: false);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorFitInterceptJsonKey], false);
    });

    test('should serialize interceptScale field, '
        'interceptScale=$interceptScale1', () {
      final classifier = createClassifier(interceptScale: interceptScale1);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorInterceptScaleJsonKey], interceptScale1);
    });

    test('should serialize interceptScale field, '
        'interceptScale=$interceptScale2', () {
      final classifier = createClassifier(interceptScale: interceptScale2);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorInterceptScaleJsonKey], interceptScale2);
    });

    test('should serialize interceptScale field, '
        'interceptScale=$interceptScale3', () {
      final classifier = createClassifier(interceptScale: interceptScale3);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorInterceptScaleJsonKey], interceptScale3);
    });
    
    test('should serialize coefficientsByClasses field', () {
      final classifier = createClassifier();
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorCoefficientsByClassesJsonKey],
          matrixToJson(classifier.coefficientsByClasses));
    });

    test('should serialize dtype field, dtype=DType.float32', () {
      final classifier = createClassifier(dtype: DType.float32);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorDTypeJsonKey],
          dTypeToJson(DType.float32));
    });

    test('should serialize dtype field, dtype=DType.float64', () {
      final classifier = createClassifier(dtype: DType.float64);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorDTypeJsonKey],
          dTypeToJson(DType.float64));
    });

    test('should serialize linkFunction field, dtype=DType.float32', () {
      final classifier = createClassifier(dtype: DType.float32);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorLinkFunctionJsonKey],
          softmaxLinkFunctionEncoded);
    });

    test('should serialize linkFunction field, dtype=DType.float64', () {
      final classifier = createClassifier(dtype: DType.float64);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorLinkFunctionJsonKey],
          softmaxLinkFunctionEncoded);
    });

    test('should serialize positiveLabel field, '
        'positiveLabel=$positiveLabel1', () {
      final classifier = createClassifier(positiveLabel: positiveLabel1);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorPositiveLabelJsonKey], positiveLabel1);
    });

    test('should serialize positiveLabel field, '
        'positiveLabel=$positiveLabel2', () {
      final classifier = createClassifier(positiveLabel: positiveLabel2);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorPositiveLabelJsonKey], positiveLabel2);
    });

    test('should serialize positiveLabel field, '
        'positiveLabel=$positiveLabel3', () {
      final classifier = createClassifier(positiveLabel: positiveLabel3);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorPositiveLabelJsonKey], positiveLabel3);
    });

    test('should serialize negativeLabel field, '
        'negativeLabel=$negativeLabel1', () {
      final classifier = createClassifier(negativeLabel: negativeLabel1);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorNegativeLabelJsonKey], negativeLabel1);
    });

    test('should serialize negativeLabel field, '
        'positiveLabel=$negativeLabel2', () {
      final classifier = createClassifier(negativeLabel: negativeLabel2);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorNegativeLabelJsonKey], negativeLabel2);
    });

    test('should serialize negativeLabel field, '
        'negativeLabel=$negativeLabel3', () {
      final classifier = createClassifier(negativeLabel: negativeLabel3);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorNegativeLabelJsonKey], negativeLabel3);
    });

    test('should serialize schemaVersion field', () {
      final classifier = createClassifier();
      final serialized = classifier.toJson();

      expect(serialized[jsonSchemaVersionJsonKey],
          softmaxRegressorJsonSchemaVersion);
    });
  });

  group('SoftmaxRegressor.saveAsJson', () {
    tearDown(() async {
      final file = File(fileName);

      if (await file.exists()) {
        await file.delete();
      }

      await module.reset();
      await softmaxRegressorModule.reset();
    });

    test('should return a pointer to a file while saving the model into the '
        'file', () async {
      final classifier = createClassifier();
      final file = await classifier.saveAsJson(fileName);

      expect(await file.exists(), isTrue);
      expect(await file.path, fileName);
    });

    test('should save the model to a file as json, '
        'dtype=DType.float32', () async {
      final classifier = createClassifier(dtype: DType.float32);
      await classifier.saveAsJson(fileName);

      final file = File(fileName);
      final decodedData = await file.readAsString();
      final restoredClassifier = SoftmaxRegressor.fromJson(decodedData);

      expect(restoredClassifier.interceptScale, classifier.interceptScale);
      expect(restoredClassifier.fitIntercept, classifier.fitIntercept);
      expect(restoredClassifier.targetNames, classifier.targetNames);
      expect(restoredClassifier.coefficientsByClasses,
          classifier.coefficientsByClasses);
      expect(restoredClassifier.linkFunction.runtimeType,
          classifier.linkFunction.runtimeType);
      expect(restoredClassifier.dtype, classifier.dtype);
    });

    test('should save the model to a file as json, '
        'dtype=DType.float64', () async {
      final classifier = createClassifier(dtype: DType.float64);
      await classifier.saveAsJson(fileName);

      final file = File(fileName);
      final decodedData = await file.readAsString();
      final restoredClassifier = SoftmaxRegressor.fromJson(decodedData);

      expect(restoredClassifier.interceptScale, classifier.interceptScale);
      expect(restoredClassifier.fitIntercept, classifier.fitIntercept);
      expect(restoredClassifier.targetNames, classifier.targetNames);
      expect(restoredClassifier.coefficientsByClasses,
          classifier.coefficientsByClasses);
      expect(restoredClassifier.linkFunction.runtimeType,
          classifier.linkFunction.runtimeType);
      expect(restoredClassifier.dtype, classifier.dtype);
    });

    test('should save the model to a file as json, '
        'collectLearningData=false', () async {
      final classifier = createClassifier(
        dtype: DType.float32,
        collectLearningData: false,
      );
      await classifier.saveAsJson(fileName);

      final file = File(fileName);
      final decodedData = await file.readAsString();
      final restoredClassifier = SoftmaxRegressor.fromJson(decodedData);

      expect(restoredClassifier.costPerIteration, isNull);
    });

    test('should save the model to a file as json, '
        'collectLearningData=true', () async {
      final classifier = createClassifier(
        dtype: DType.float32,
        collectLearningData: true,
      );
      await classifier.saveAsJson(fileName);

      final file = File(fileName);
      final decodedData = await file.readAsString();
      final restoredClassifier = SoftmaxRegressor.fromJson(decodedData);

      expect(restoredClassifier.costPerIteration, classifier.costPerIteration);
    });
  });
}
