import 'dart:io';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/_init_module.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_json_keys.dart';
import 'package:ml_algo/src/di/injector.dart';
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

  final createClassifier = ({
    bool fitIntercept = true,
    double interceptScale = 10,
    DType dtype = DType.float32,
    num positiveLabel = 1,
    num negativeLabel = -1,
    bool collectLearningData = false,
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
      fitIntercept: fitIntercept,
      interceptScale: interceptScale,
      positiveLabel: positiveLabel,
      negativeLabel: negativeLabel,
      collectLearningData: collectLearningData,
      dtype: dtype,
    );
  };

  group('SoftmaxRegressor.toJson', () {
    tearDown(() {
      injector.clearAll();
      softmaxRegressorInjector.clearAll();
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
          float32SoftmaxLinkFunctionEncoded);
    });

    test('should serialize linkFunction field, dtype=DType.float64', () {
      final classifier = createClassifier(dtype: DType.float64);
      final serialized = classifier.toJson();

      expect(serialized[softmaxRegressorLinkFunctionJsonKey],
          float64SoftmaxLinkFunctionEncoded);
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
  });

  group('SoftmaxRegressor.saveAsJson', () {
    tearDown(() async {
      final file = File(fileName);

      if (await file.exists()) {
        await file.delete();
      }

      injector.clearAll();
      softmaxRegressorInjector.clearAll();
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
