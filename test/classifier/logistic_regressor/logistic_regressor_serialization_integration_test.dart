import 'dart:io';

import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_json_keys.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/link_function/link_function_encoded_values.dart';
import 'package:ml_algo/src/link_function/logit/float32_inverse_logit_function.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/dtype_to_json.dart';
import 'package:ml_linalg/matrix_to_json.dart';
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
  final positiveLabel3 = 0;

  final negativeLabel1 = 100;
  final negativeLabel2 = -100;
  final negativeLabel3 = 0;

  final createClassifier = ({
    String targetName = 'col_3',
    bool fitIntercept = false,
    double interceptScale = 3.0,
    double probabilityThreshold = 0.9,
    num positiveLabel = 1,
    num negativeLabel = 0,
    DType dtype = DType.float32,
  }) => LogisticRegressor(
    samples,
    targetName,
    iterationsLimit: 2,
    learningRateType: LearningRateType.constant,
    initialLearningRate: 1.0,
    batchSize: 5,
    fitIntercept: fitIntercept,
    interceptScale: interceptScale,
    dtype: dtype,
    probabilityThreshold: probabilityThreshold,
    positiveLabel: positiveLabel,
    negativeLabel: negativeLabel,
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

    test('should serialize interceptScale field, '
        'interceptScale=$interceptScale1', () {
      final classifier = createClassifier(interceptScale: interceptScale1);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorInterceptScaleJsonKey],
        interceptScale1,
      );
    });

    test('should serialize interceptScale field, '
        'interceptScale=$interceptScale2', () {
      final classifier = createClassifier(interceptScale: interceptScale2);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorInterceptScaleJsonKey],
        interceptScale2,
      );
    });

    test('should serialize interceptScale field, '
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

    test('should serialize probabilityThreshold field, '
        'value=$probabilityThreshold1', () {
      final classifier = createClassifier(
          probabilityThreshold: probabilityThreshold1);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorProbabilityThresholdJsonKey],
        probabilityThreshold1,
      );
    });

    test('should serialize probabilityThreshold field, '
        'value=$probabilityThreshold2', () {
      final classifier = createClassifier(
          probabilityThreshold: probabilityThreshold2);
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

    test('should serialize linkFunction field, dtype=DType.float32', () {
      final classifier = createClassifier(dtype: DType.float32);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorLinkFunctionJsonKey],
        float32InverseLogitLinkFunctionEncoded,
      );
    });

    test('should serialize linkFunction field, dtype=DType.float64', () {
      final classifier = createClassifier(dtype: DType.float64);
      final serialized = classifier.toJson();

      expect(
        serialized[logisticRegressorLinkFunctionJsonKey],
        float64InverseLogitLinkFunctionEncoded,
      );
    });
  });

  group('LogisticRegressor.saveAsJson', () {
    tearDown(() async {
      final file = File(fileName);

      if (await file.exists()) {
        await file.delete();
      }
    });

    test('should return a pointer to a json file while saving serialized '
        'data', () async {
      final classifier = createClassifier();
      final file = await classifier.saveAsJson(fileName);

      expect(await file.exists(), isTrue);
      expect(file.path, fileName);
    });

    test('should restore a classifier instance from json file', () async {
      final classifier = createClassifier();
      await classifier.saveAsJson(fileName);

      final file = File(fileName);
      final encodedData = await file.readAsString();
      final restoredClassifier = LogisticRegressor.fromJson(encodedData);

      expect(restoredClassifier.coefficientsByClasses,
          classifier.coefficientsByClasses);
      expect(restoredClassifier.interceptScale, classifier.interceptScale);
      expect(restoredClassifier.fitIntercept, classifier.fitIntercept);
      expect(restoredClassifier.dtype, classifier.dtype);
      expect(restoredClassifier.linkFunction,
          isA<Float32InverseLogitLinkFunction>());
      expect(restoredClassifier.classNames, [targetName]);
    });
  });
}
