import 'package:ml_algo/src/classifier/logistic_regressor/gradient_logistic_regressor.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

import '../test_utils/helpers/floating_point_iterable_matchers.dart';

void main() {
  final firstClass = [1.0];
  final secondClass = [0.0];
  final thirdClass = [0.0];

  group('Logistic regressor', () {
    test('should extract class labels from the test_data', () {
      final features = Matrix.fromList([
        [5.0, 7.0, 6.0],
        [1.0, 2.0, 3.0],
        [10.0, 12.0, 31.0],
        [9.0, 8.0, 5.0],
        [4.0, 0.0, 1.0],
        [4.0, 0.0, 1.0],
        [4.0, 0.0, 1.0],
      ]);
      final labels = Matrix.fromList([
        [3.0],
        [1.0],
        [3.0],
        [2.0],
        [2.0],
        [0.0],
        [0.0],
      ]);

      final classifier = GradientLogisticRegressor(
          features, labels,
          iterationsLimit: 2,
          learningRateType: LearningRateType.constant,
          initialLearningRate: 1.0,
          fitIntercept: false
      );

      expect(classifier.classLabels, equals([
        [3.0],
        [1.0],
        [2.0],
        [0.0],
      ]));
    });

    test('should properly fit given test_data', () {
      final features = Matrix.fromList([
        [5.0, 7.0, 6.0],
        [1.0, 2.0, 3.0],
        [10.0, 12.0, 31.0],
        [9.0, 8.0, 5.0],
        [4.0, 0.0, 1.0],
      ]);
      final labels = Matrix.fromList([
        firstClass,
        secondClass,
        secondClass,
        thirdClass,
        firstClass,
      ]);

      final classifier = GradientLogisticRegressor(
          features, labels,
          iterationsLimit: 2,
          learningRateType: LearningRateType.constant,
          initialLearningRate: 1.0,
          batchSize: 5,
          fitIntercept: false
      );

      expect(classifier.weightsByClasses, matrixAlmostEqualTo([
        [3.5,],
        [-0.5,],
        [-9.0,],
      ], 1e-2));
    });

    test('should make prediction', () {
      final features = Matrix.fromList([
        [5.0, 7.0, 6.0],
        [1.0, 2.0, 3.0],
        [10.0, 12.0, 31.0],
        [9.0, 8.0, 5.0],
        [4.0, 0.0, 1.0],
      ]);

      final labels = Matrix.fromList([
        firstClass,
        secondClass,
        secondClass,
        thirdClass,
        firstClass,
      ]);

      final classifier = GradientLogisticRegressor(
          features, labels,
          iterationsLimit: 2,
          learningRateType: LearningRateType.constant,
          initialLearningRate: 1.0,
          batchSize: 5,
          fitIntercept: false
      );

      final newFeatures = Matrix.fromList([
        [2.0, 4.0, 1.0],
      ]);
      final probabilities = classifier.predictProbabilities(newFeatures);
      final classes = classifier.predict(newFeatures);

      expect(probabilities, equals([[0.01798621006309986]]));
      expect(classes, equals([thirdClass]));
    });

    test('should evaluate prediction quality, accuracy = 0', () {
      final features = Matrix.fromList([
        [5.0, 7.0, 6.0],
        [1.0, 2.0, 3.0],
        [10.0, 12.0, 31.0],
        [9.0, 8.0, 5.0],
        [4.0, 0.0, 1.0],
      ]);
      final labels = Matrix.fromList([
        firstClass,
        secondClass,
        secondClass,
        thirdClass,
        firstClass,
      ]);

      final classifier = GradientLogisticRegressor(
          features, labels,
          iterationsLimit: 2,
          learningRateType: LearningRateType.constant,
          initialLearningRate: 1.0,
          batchSize: 5,
          fitIntercept: false
      );

      final newFeatures = Matrix.fromList([
        [2.0, 4.0, 1.0],
      ]);
      final origLabels = Matrix.fromList([
        [1.0]
      ]);
      final score =
          classifier.test(newFeatures, origLabels, MetricType.accuracy);
      expect(score, equals(0.0));
    });

    test('should evaluate prediction quality, accuracy = 1', () {
      final features = Matrix.fromList([
        [5.0, 7.0, 6.0],
        [1.0, 2.0, 3.0],
        [10.0, 12.0, 31.0],
        [9.0, 8.0, 5.0],
        [4.0, 0.0, 1.0],
      ]);
      final labels = Matrix.fromList([
        firstClass,
        secondClass,
        secondClass,
        thirdClass,
        firstClass,
      ]);

      final classifier = GradientLogisticRegressor(
          features, labels,
          iterationsLimit: 2,
          learningRateType: LearningRateType.constant,
          initialLearningRate: 1.0,
          batchSize: 5,
          fitIntercept: false
      );

      final newFeatures = Matrix.fromList([
        [2.0, 4.0, 1.0],
      ]);
      final newLabels = Matrix.fromList([
        thirdClass,
      ]);
      final score =
          classifier.test(newFeatures, newLabels, MetricType.accuracy);
      expect(score, equals(1.0));
    });

    test('should consider intercept term', () {
      final features = Matrix.fromList([
        [5.0, 7.0, 6.0],
        [1.0, 2.0, 3.0],
      ]);
      final labels = Matrix.fromList([
        [1.0],
        [0.0],
      ]);
      final classifier = GradientLogisticRegressor(
          features, labels,
          iterationsLimit: 1,
          learningRateType: LearningRateType.constant,
          initialLearningRate: 1.0,
          batchSize: 2,
          fitIntercept: true
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
      expect(classifier.weightsByClasses, equals([
        [0.0],
        [2.0],
        [2.5],
        [1.5],
      ]));
    });

    test('should consider intercept scale if intercept term is going to be '
        'fitted', () {
      final features = Matrix.fromList([
        [5.0, 7.0, 6.0],
        [1.0, 2.0, 3.0],
        [3.0, 4.0, 5.0],
      ]);
      final labels = Matrix.fromList([
        [1.0],
        [0.0],
        [1.0],
      ]);

      final classifier = GradientLogisticRegressor(
          features, labels,
          iterationsLimit: 1,
          learningRateType: LearningRateType.constant,
          initialLearningRate: 1.0,
          batchSize: 3,
          fitIntercept: true,
          interceptScale: 2.0
      );

      // as the intercept is required to be fitted, our test_data should look as follows:
      //
      // [5.0, 7.0, 6.0] => [2.0, 5.0, 7.0, 6.0]
      // [1.0, 2.0, 3.0] => [2.0, 1.0, 2.0, 3.0]
      // [3.0, 4.0, 5.0] => [2.0, 3.0, 4.0, 5.0]
      //
      // we add a new column to the test_data, the column is consisted of just ones, so we considered that our fitted line will
      // start not right at the origin, but at the origin + intercept equal to 2.0. What is the value of the intercept? To answer
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
      expect(classifier.weightsByClasses, equals([
        [1.0],
        [3.5],
        [4.5],
        [4.0],
      ]));
    });
  });
}
