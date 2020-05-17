import 'dart:io';

import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_json_keys.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/link_function/link_function_encoded_values.dart';
import 'package:ml_algo/src/link_function/logit/float32_inverse_logit_function.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_tech/unit_testing/matchers/iterable_2d_almost_equal_to.dart';
import 'package:test/test.dart';

void main() {
  group('Logistic regressor', () {
    final data = <Iterable<num>>[
      [5.0, 7.0, 6.0, 1.0],
      [1.0, 2.0, 3.0, 0.0],
      [10.0, 12.0, 31.0, 0.0],
      [9.0, 8.0, 5.0, 0.0],
      [4.0, 0.0, 1.0, 1.0],
    ];
    final targetName = 'col_3';
    final samples = DataFrame(data, headerExists: false);

    tearDownAll(() => injector = null);

    test('should fit given data', () {
      final classifier = LogisticRegressor(
        samples,
        targetName,
        iterationsLimit: 2,
        learningRateType: LearningRateType.constant,
        initialLearningRate: 1.0,
        batchSize: 5,
        fitIntercept: false,
      );

      expect(classifier.coefficientsByClasses, iterable2dAlmostEqualTo([
        [3.5,],
        [-0.5,],
        [-9.0,],
      ], 1e-2));
    });

    test('should make prediction', () {
      final classifier = LogisticRegressor(
        samples,
        targetName,
        iterationsLimit: 2,
        learningRateType: LearningRateType.constant,
        initialLearningRate: 1.0,
        batchSize: 5,
        fitIntercept: false,
      );

      final newFeatures = Matrix.fromList([
        [2.0, 4.0, 1.0],
      ]);

      final probabilities = classifier.predictProbabilities(
        DataFrame.fromMatrix(newFeatures),
      );

      final classes = classifier.predict(
        DataFrame.fromMatrix(newFeatures),
      );

      expect(probabilities.header, equals(['col_3']));
      expect(probabilities.toMatrix(), equals([[0.01798621006309986]]));
      expect(classes.header, equals(['col_3']));
      expect(classes.toMatrix(), equals([[0.0]]));
    });

    test('should evaluate prediction quality, accuracy = 0', () {
      final classifier = LogisticRegressor(
          samples,
          targetName,
          iterationsLimit: 2,
          learningRateType: LearningRateType.constant,
          initialLearningRate: 1.0,
          batchSize: 5,
          fitIntercept: false
      );

      final newSamples = DataFrame([
        <num>[2.0, 4.0, 1.0, 1.0],
      ], header: ['first', 'second', 'third', 'target'], headerExists: false);

      final score = classifier.assess(newSamples, ['target'],
          MetricType.accuracy);

      expect(score, equals(0.0));
    });

    test('should evaluate prediction quality, accuracy = 1', () {
      final classifier = LogisticRegressor(
          samples,
          targetName,
          iterationsLimit: 2,
          learningRateType: LearningRateType.constant,
          initialLearningRate: 1.0,
          batchSize: 5,
          fitIntercept: false
      );

      final newFeatures = DataFrame([
        <num>[2, 4, 1, 0],
      ], header: ['first', 'second', 'third', 'target'], headerExists: false);

      final score = classifier.assess(newFeatures, ['target'],
          MetricType.accuracy);

      expect(score, equals(1.0));
    });

    test('should consider intercept term', () {
      final features = DataFrame(<Iterable<num>>[
        [5.0, 7.0, 6.0, 1.0],
        [1.0, 2.0, 3.0, 0.0],
      ], headerExists: false);

      final classifier = LogisticRegressor(
        features,
        targetName,
        iterationsLimit: 1,
        learningRateType: LearningRateType.constant,
        initialLearningRate: 1.0,
        batchSize: 2,
        fitIntercept: true,
      );
      // as the intercept is required to be fitted, our test_data should look as follows:
      //
      // [5.0, 7.0, 6.0] => [1.0, 5.0, 7.0, 6.0]
      // [1.0, 2.0, 3.0] => [1.0, 1.0, 2.0, 3.0]
      //
      // we add a new column to the test_data, the column is consisted of just ones, so we considered that our fitted line will
      // start not right at the origin, but at the origin + some intercept. What is the value of the intercept? To answer
      // this question, we have to find out the intercept's weight
      //
      // given test_data
      // ----------------------------------------------
      // | X (features):           | Y (class labels):|
      // ---------------------------------------------|
      // | [1.0, 5.0, 7.0, 6.0]    | [0.0]            |
      // | [1.0, 1.0, 2.0, 3.0]    | [1.0]            |
      // ----------------------------------------------
      //
      // formula for derivative:
      // sum(x_i_j * (indicator(y=1) - P(y=1|x_i,w)))
      //
      // formula for the update:
      // w_new = w_prev + eta * derivative (gradient ascent)
      //
      // ===============================================================================================================
      // ITERATION 1, weights = [0.0, 0.0, 0.0, 0.0]
      // ===============================================================================================================
      // weights for class label 0.0:
      // 1.0 * (1 - (1 / (1 + e^0))) = 0.5
      // 5.0 * (1 - (1 / (1 + e^0))) = 2.5
      // 7.0 * (1 - (1 / (1 + e^0))) = 3.5
      // 6.0 * (1 - (1 / (1 + e^0))) = 3.0
      //
      // 1.0 * (0 - (1 / (1 + e^0))) = -0.5
      // 1.0 * (0 - (1 / (1 + e^0))) = -0.5
      // 2.0 * (0 - (1 / (1 + e^0))) = -1.0
      // 3.0 * (0 - (1 / (1 + e^0))) = -1.5
      //
      // derivative: [0.0, 2.0, 2.5, 1.5]
      // update: [0.0, 0.0, 0.0, 0.0] + 1.0 * [0.0, 2.0, 2.5, 1.5] = [0.0, 2.0, 2.5, 1.5]
      //
      // weights for class label 1.0:
      // 1.0 * (0 - (1 / (1 + e^0))) = -0.5
      // 5.0 * (0 - (1 / (1 + e^0))) = -2.5
      // 7.0 * (0 - (1 / (1 + e^0))) = -3.5
      // 6.0 * (0 - (1 / (1 + e^0))) = -3.0
      //
      // 1.0 * (1 - (1 / (1 + e^0))) = 0.5
      // 1.0 * (1 - (1 / (1 + e^0))) = 0.5
      // 2.0 * (1 - (1 / (1 + e^0))) = 1.0
      // 3.0 * (1 - (1 / (1 + e^0))) = 1.5
      //
      // derivative: [0.0, -2.0, -2.5, -1.5]
      // update: [0.0, 0.0, 0.0, 0.0] + 1.0 * [0.0, -2.0, -2.5, -1.5] = [0.0, -2.0, -2.5, -1.5]
      expect(classifier.coefficientsByClasses, equals([
        [0.0],
        [2.0],
        [2.5],
        [1.5],
      ]));
    });

    test('should consider intercept scale if intercept term is going to be '
        'fitted', () {
      final samples = DataFrame(<Iterable<num>>[
        [5.0, 7.0, 6.0, 1.0],
        [1.0, 2.0, 3.0, 0.0],
        [3.0, 4.0, 5.0, 1.0],
      ], headerExists: false);

      final classifier = LogisticRegressor(
        samples,
        targetName,
        iterationsLimit: 1,
        learningRateType: LearningRateType.constant,
        initialLearningRate: 1.0,
        batchSize: 3,
        fitIntercept: true,
        interceptScale: 2.0,
      );

      // as the intercept is required to be fitted, our test_data should look as follows:
      //
      // [5.0, 7.0, 6.0] => [2.0, 5.0, 7.0, 6.0]
      // [1.0, 2.0, 3.0] => [2.0, 1.0, 2.0, 3.0]
      // [3.0, 4.0, 5.0] => [2.0, 3.0, 4.0, 5.0]
      //
      // we add a new column to the test_data, the column is consisted of just ones, so we considered that our fitted line will
      // start not right from the origin, but from the origin + intercept equal to 2.0. What is the value of the intercept? To answer
      // this question, we have to find out the intercept's weight
      //
      // given test_data
      // ----------------------------------------------
      // | X (features):           | Y (class labels):|
      // ---------------------------------------------|
      // | [2.0, 5.0, 7.0, 6.0]    | [0.0]            |
      // | [2.0, 1.0, 2.0, 3.0]    | [1.0]            |
      // | [2.0, 3.0, 4.0, 5.0]    | [0.0]            |
      // ----------------------------------------------
      //
      // formula for derivative:
      // sum(x_i_j * (indicator(y=1) - P(y=1|x_i,w)))
      //
      // formula for the update:
      // w_new = w_prev + eta * derivative (gradient ascent)
      //
      // ===============================================================================================================
      // ITERATION 1, weights = [0.0, 0.0, 0.0, 0.0]
      // ===============================================================================================================
      // weights for class label 0.0:
      // 2.0 * (1 - (1 / (1 + e^0))) = 1.0
      // 5.0 * (1 - (1 / (1 + e^0))) = 2.5
      // 7.0 * (1 - (1 / (1 + e^0))) = 3.5
      // 6.0 * (1 - (1 / (1 + e^0))) = 3.0
      //
      // 2.0 * (0 - (1 / (1 + e^0))) = -1.0
      // 1.0 * (0 - (1 / (1 + e^0))) = -0.5
      // 2.0 * (0 - (1 / (1 + e^0))) = -1.0
      // 3.0 * (0 - (1 / (1 + e^0))) = -1.5
      //
      // 2.0 * (1 - (1 / (1 + e^0))) = 1.0
      // 3.0 * (1 - (1 / (1 + e^0))) = 1.5
      // 4.0 * (1 - (1 / (1 + e^0))) = 2.0
      // 5.0 * (1 - (1 / (1 + e^0))) = 2.5
      //
      // derivative: [1.0, 3.5, 4.5, 4.0]
      // update: [0.0, 0.0, 0.0, 0.0] + 1.0 * [1.0, 3.5, 4.5, 4.0] = [1.0, 3.5, 4.5, 4.0]
      //
      // weights for class label 1.0:
      // 2.0 * (0 - (1 / (1 + e^0))) = -1.0
      // 5.0 * (0 - (1 / (1 + e^0))) = -2.5
      // 7.0 * (0 - (1 / (1 + e^0))) = -3.5
      // 6.0 * (0 - (1 / (1 + e^0))) = -3.0
      //
      // 2.0 * (1 - (1 / (1 + e^0))) = 1.0
      // 1.0 * (1 - (1 / (1 + e^0))) = 0.5
      // 2.0 * (1 - (1 / (1 + e^0))) = 1.0
      // 3.0 * (1 - (1 / (1 + e^0))) = 1.5
      //
      // 2.0 * (0 - (1 / (1 + e^0))) = -1.0
      // 3.0 * (0 - (1 / (1 + e^0))) = -1.5
      // 4.0 * (0 - (1 / (1 + e^0))) = -2.0
      // 5.0 * (0 - (1 / (1 + e^0))) = -2.5
      //
      // derivative: [-1.0, -3.5, -4.5, -4.0]
      // update: [0.0, 0.0, 0.0, 0.0] + 1.0 * [-1.0, -3.5, -4.5, -4.0] = [-1.0, -3.5, -4.5, -4.0]
      expect(classifier.coefficientsByClasses, equals([
        [1.0],
        [3.5],
        [4.5],
        [4.0],
      ]));
    });

    group('serialization', () {
      final fitIntercept = false;
      final interceptScale = 3.0;
      final dType = DType.float32;
      final probabilityThreshold = .9;
      final positiveLabel = 1;
      final negativeLabel = 0;

      final fileName = 'test/classifier/logistic_regressor/logistic_regressor.json';

      LogisticRegressor classifier;

      setUp(() {
        classifier = LogisticRegressor(
          samples,
          targetName,
          iterationsLimit: 2,
          learningRateType: LearningRateType.constant,
          initialLearningRate: 1.0,
          batchSize: 5,
          fitIntercept: fitIntercept,
          interceptScale: interceptScale,
          dtype: dType,
          probabilityThreshold: probabilityThreshold,
          positiveLabel: positiveLabel,
        );
      });

      tearDown(() async {
        final file = File(fileName);

        if (await file.exists()) {
          await file.delete();
        }
      });

      test('should serialize', () {
        final serialized = classifier.toJson();

        expect(serialized, {
          logisticRegressorCoefficientsByClassesJsonKey: matrixToJson(
              classifier.coefficientsByClasses),
          logisticRegressorClassNamesJsonKey: [targetName],
          logisticRegressorFitInterceptJsonKey: fitIntercept,
          logisticRegressorInterceptScaleJsonKey: interceptScale,
          logisticRegressorDTypeJsonKey: dTypeToJson(dType),
          logisticRegressorProbabilityThresholdJsonKey: probabilityThreshold,
          logisticRegressorPositiveLabelJsonKey: positiveLabel,
          logisticRegressorNegativeLabelJsonKey: negativeLabel,
          logisticRegressorLinkFunctionJsonKey: float32InverseLogitLinkFunctionEncoded,
        });
      });

      test('should return a pointer to a json file while saving serialized '
          'data', () async {
        final file = await classifier.saveAsJson(fileName);

        expect(await file.exists(), isTrue);
      });

      test('should restore a classifier instance from json file', () async {
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
  });
}
