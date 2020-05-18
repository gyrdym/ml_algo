import 'dart:io';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_json_keys.dart';
import 'package:ml_algo/src/link_function/link_function_encoded_values.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

void main() {
  group('SoftmaxRegressor', () {
    final featureNames = ['feature 1', 'feature 2', 'feature 3'];
    final targetNames = ['class_1', 'class_2', 'class_3'];
    final positiveLabel = 100;
    final negativeLabel = -100;
    final sourceData = <Iterable<dynamic>>[
      <String>[...featureNames, ...targetNames],
      <num>[   100,    200, 300.89, positiveLabel, negativeLabel, negativeLabel],
      <num>[   444,   20.7, 300.89, negativeLabel, negativeLabel, positiveLabel],
      <num>[   100, -20000, -0.003, negativeLabel, negativeLabel, positiveLabel],
      <num>[   100,    200,    1e5, negativeLabel, positiveLabel, negativeLabel],
      <num>[-0.874, 932.12,   0.98, positiveLabel, negativeLabel, negativeLabel],
    ];
    final dataFrame = DataFrame(sourceData, headerExists: true);
    final fitIntercept = true;
    final interceptScale = 123.2;
    final dtype = DType.float32;
    final fileName = 'test/classifier/softmax_regressor/softmax_regressor.json';
    SoftmaxRegressor classifier;

    setUp(() {
      classifier = SoftmaxRegressor(
        dataFrame,
        targetNames,
        fitIntercept: fitIntercept,
        interceptScale: interceptScale,
        positiveLabel: positiveLabel,
        negativeLabel: negativeLabel,
        dtype: dtype,
      );
    });

    tearDown(() async {
      final file = File(fileName);

      if (await file.exists()) {
        await file.delete();
      }
    });

    test('should return a json-serializable map', () {
      final map = classifier.toJson();

      expect(map, {
        softmaxRegressorClassNamesJsonKey: targetNames,
        softmaxRegressorFitInterceptJsonKey: fitIntercept,
        softmaxRegressorInterceptScaleJsonKey: interceptScale,
        softmaxRegressorCoefficientsByClassesJsonKey: matrixToJson(
            classifier.coefficientsByClasses),
        softmaxRegressorDTypeJsonKey: dTypeToJson(dtype),
        softmaxRegressorLinkFunctionJsonKey: float32SoftmaxLinkFunctionEncoded,
        softmaxRegressorPositiveLabelJsonKey: positiveLabel,
        softmaxRegressorNegativeLabelJsonKey: negativeLabel,
      });
    });

    test('should return a pointer to a file while saving the model into the '
        'file', () async {
      final file = await classifier.saveAsJson(fileName);

      expect(await file.exists(), isTrue);
    });

    test('should save the model to a file as json', () async {
      await classifier.saveAsJson(fileName);

      final file = File(fileName);
      final decodedData = await file.readAsString();
      final restoredClassifier = SoftmaxRegressor.fromJson(decodedData);

      expect(restoredClassifier.interceptScale, classifier.interceptScale);
      expect(restoredClassifier.fitIntercept, classifier.fitIntercept);
      expect(restoredClassifier.classNames, classifier.classNames);
      expect(restoredClassifier.coefficientsByClasses,
          classifier.coefficientsByClasses);
      expect(restoredClassifier.linkFunction.runtimeType,
          classifier.linkFunction.runtimeType);
      expect(restoredClassifier.dtype, classifier.dtype);
    });
  });
}
