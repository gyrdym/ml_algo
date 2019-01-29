import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/metric_type.dart';
import 'package:ml_algo/src/classifier/logistic_regressor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

void main() {
  LogisticRegressor classifier;

  group('Logistic regressor', () {
    setUp(() {
      classifier = LogisticRegressor(batchSize: 5, iterationLimit: 2, learningRateType: LearningRateType.constant,
          learningRate: 1.0);
    });

    test('should extract class labels from the test_data', () {
      final features = MLMatrix.from([
        [5.0, 7.0, 6.0],
        [1.0, 2.0, 3.0],
        [10.0, 12.0, 31.0],
        [9.0, 8.0, 5.0],
        [4.0, 0.0, 1.0],
        [4.0, 0.0, 1.0],
        [4.0, 0.0, 1.0],
      ]);
      final labels = MLVector.from([3.0, 1.0, 3.0, 2.0, 2.0, 0.0, 0.0]);
      classifier.fit(features, labels);

      expect(classifier.classLabels, equals([3.0, 1.0, 2.0, 0.0]));
    });

    test('should properly fit given test_data', () {
      final features = MLMatrix.from([
        [5.0, 7.0, 6.0],
        [1.0, 2.0, 3.0],
        [10.0, 12.0, 31.0],
        [9.0, 8.0, 5.0],
        [4.0, 0.0, 1.0],
      ]);
      final labels = MLVector.from([0.0, 1.0, 1.0, 2.0, 0.0]);
      classifier.fit(features, labels);

      // given test_data
      // -----------------------------------------
      // | X (features):      | Y (class labels):|
      // ----------------------------------------|
      // | [5.0, 7.0, 6.0]    | [0.0]            |
      // | [1.0, 2.0, 3.0]    | [1.0]            |
      // | [10.0, 12.0, 31.0] | [1.0]            |
      // | [9.0, 8.0, 5.0]    | [2.0]            |
      // | [4.0, 0.0, 1.0]    | [0.0]            |
      // -----------------------------------------
      //
      // formula for derivative:
      // sum(x_i_j * (indicator(y=1) - P(y=1|x_i,w)))
      //
      // formula for the update:
      // w_new = w_prev + eta * derivative (gradient ascent)
      //
      // ===============================================================================================================
      // ITERATION 1
      // ===============================================================================================================
      //
      // weights for class 0.0:
      // current weights: [0.0, 0.0, 0.0]
      // 5.0 * (1 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[5.0, 7.0, 6.0])))) - dot sign means dot product
      // 7.0 * (1 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[5.0, 7.0, 6.0]))))
      // 6.0 * (1 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[5.0, 7.0, 6.0]))))
      // -----------------------------------------------------------------
      // [2.5, 3.5, 3.0]
      //
      // 1.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[1.0, 2.0, 3.0]))))
      // 2.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[1.0, 2.0, 3.0]))))
      // 3.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[1.0, 2.0, 3.0]))))
      //-----------------------------------------------------------------
      // [-0.5, -1.0, -1.5]
      //
      // 10.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[10.0, 12.0, 31.0]))))
      // 12.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[10.0, 12.0, 31.0]))))
      // 31.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[10.0, 12.0, 31.0]))))
      //-----------------------------------------------------------------
      // [-5.0, -6.0, -15.5]
      //
      // 9.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[9.0, 8.0, 5.0]))))
      // 8.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[9.0, 8.0, 5.0]))))
      // 5.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[9.0, 8.0, 5.0]))))
      //-----------------------------------------------------------------
      // [-4.5, -4.0, -2.5]
      //
      // 4.0 * (1 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[4.0, 0.0, 1.0]))))
      // 0.0 * (1 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[4.0, 0.0, 1.0]))))
      // 1.0 * (1 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[4.0, 0.0, 1.0]))))
      //-----------------------------------------------------------------
      // [2.0, 0.0, 0.5]
      //
      // derivative (sum up all the vectors above):
      // [-5.5, -7.5, -16.0]
      //
      // update:
      // [0.0, 0.0, 0.0] + eta * [-5.5, -7.5, -16.0] = [0.0, 0.0, 0.0] + 1.0 * [-5.5, -7.5, -16.0] = [-5.5, -7.5, -16.0]
      //
      // weights for class 1.0:
      // current weights: [0.0, 0.0, 0.0]
      // 5.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[5.0, 7.0, 6.0])))) - dot sign means dot product
      // 7.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[5.0, 7.0, 6.0]))))
      // 6.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[5.0, 7.0, 6.0]))))
      //-----------------------------------------------------------------
      // [-2.5, -3.5, -3.0]
      //
      // 1.0 * (1 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[1.0, 2.0, 3.0]))))
      // 2.0 * (1 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[1.0, 2.0, 3.0]))))
      // 3.0 * (1 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[1.0, 2.0, 3.0]))))
      //-----------------------------------------------------------------
      // [0.5, 1.0, 1.5]
      //
      // 10.0 * (1 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[10.0, 12.0, 31.0]))))
      // 12.0 * (1 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[10.0, 12.0, 31.0]))))
      // 31.0 * (1 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[10.0, 12.0, 31.0]))))
      //-----------------------------------------------------------------
      // [5.0, 6.0, 15.5]
      //
      // 9.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[9.0, 8.0, 5.0]))))
      // 8.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[9.0, 8.0, 5.0]))))
      // 5.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[9.0, 8.0, 5.0]))))
      //-----------------------------------------------------------------
      // [-4.5, -4.0, -2.5]
      //
      // 4.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[4.0, 0.0, 1.0]))))
      // 0.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[4.0, 0.0, 1.0]))))
      // 1.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[4.0, 0.0, 1.0]))))
      //-----------------------------------------------------------------
      // [-2.0, 0.0, -0.5]
      //
      // derivative:
      // [-3.5, -0.5, 11.0]
      //
      // update:
      // [0.0, 0.0, 0.0] + eta * [-3.5, -0.5, 11.0] = [0.0, 0.0, 0.0] + 1.0 * [-3.5, -0.5, 11.0] = [-3.5, -0.5, 11.0]
      //
      // weights for class 2.0:
      // current weights: [0.0, 0.0, 0.0]
      // 5.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[5.0, 7.0, 6.0])))) - dot sign means dot product
      // 7.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[5.0, 7.0, 6.0]))))
      // 6.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[5.0, 7.0, 6.0]))))
      //-----------------------------------------------------------------
      // [-2.5, -3.5, -3.0]
      //
      // 1.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[1.0, 2.0, 3.0]))))
      // 2.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[1.0, 2.0, 3.0]))))
      // 3.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[1.0, 2.0, 3.0]))))
      //-----------------------------------------------------------------
      // [-0.5, -1.0, -1.5]
      //
      // 10.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[10.0, 12.0, 31.0]))))
      // 12.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[10.0, 12.0, 31.0]))))
      // 31.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[10.0, 12.0, 31.0]))))
      //-----------------------------------------------------------------
      // [-5.0, -6.0, -15.5]
      //
      // 9.0 * (1 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[9.0, 8.0, 5.0]))))
      // 8.0 * (1 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[9.0, 8.0, 5.0]))))
      // 5.0 * (1 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[9.0, 8.0, 5.0]))))
      //-----------------------------------------------------------------
      // [4.5, 4.0, 2.5]
      //
      // 4.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[4.0, 0.0, 1.0]))))
      // 0.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[4.0, 0.0, 1.0]))))
      // 1.0 * (0 - (1 / (1 + exp(-1 * [0.0, 0.0, 0.0].[4.0, 0.0, 1.0]))))
      //-----------------------------------------------------------------
      // [-2.0, 0.0, -0.5]
      //
      // derivative:
      // [-5.5, -6.5, -18.0]
      //
      // update:
      // [0.0, 0.0, 0.0] + eta * [-5.5, -6.5, -18.0] = [0.0, 0.0, 0.0] + 1.0 * [-5.5, -6.5, -18.0] = [-5.5, -6.5, -18.0]
      //
      // ===============================================================================================================
      // ITERATION 2
      // ===============================================================================================================
      // weights for class 0.0:
      // current weights: [-5.5, -7.5, -16.0]
      // 5.0 * (1 - (1 / (1 + exp(-1 * [-5.5, -7.5, -16.0].[5.0, 7.0, 6.0]))))
      // 7.0 * (1 - (1 / (1 + exp(-1 * [-5.5, -7.5, -16.0].[5.0, 7.0, 6.0]))))
      // 6.0 * (1 - (1 / (1 + exp(-1 * [-5.5, -7.5, -16.0].[5.0, 7.0, 6.0]))))
      // -----------------------------------------------------------------
      // [5.0, 7.0, 6.0]
      //
      // 1.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -7.5, -16.0].[1.0, 2.0, 3.0]))))
      // 2.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -7.5, -16.0].[1.0, 2.0, 3.0]))))
      // 3.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -7.5, -16.0].[1.0, 2.0, 3.0]))))
      //-----------------------------------------------------------------
      // [-3.66e-77, -7.33e-77, -1.1e-76]
      //
      // 10.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -7.5, -16.0].[10.0, 12.0, 31.0]))))
      // 12.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -7.5, -16.0].[10.0, 12.0, 31.0]))))
      // 31.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -7.5, -16.0].[10.0, 12.0, 31.0]))))
      //-----------------------------------------------------------------
      // [-3.66e-76, -4.39e-76, -1.13e-75]
      //
      // 9.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -7.5, -16.0].[9.0, 8.0, 5.0]))))
      // 8.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -7.5, -16.0].[9.0, 8.0, 5.0]))))
      // 5.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -7.5, -16.0].[9.0, 8.0, 5.0]))))
      //-----------------------------------------------------------------
      // [-3.3e-76, -2.93e-76, -1.83e-76]
      //
      // 4.0 * (1 - (1 / (1 + exp(-1 * [-5.5, -7.5, -16.0].[4.0, 0.0, 1.0]))))
      // 0.0 * (1 - (1 / (1 + exp(-1 * [-5.5, -7.5, -16.0].[4.0, 0.0, 1.0]))))
      // 1.0 * (1 - (1 / (1 + exp(-1 * [-5.5, -7.5, -16.0].[4.0, 0.0, 1.0]))))
      //-----------------------------------------------------------------
      // [4.0, 0.0, 1.0]
      //
      // derivative (sum up all the vectors above):
      // [9.0, 7.0, 7.0]
      //
      // update:
      // [-5.5, -7.5, -16.0] + eta * [9.0, 7.0, 7.0] = [-5.5, -7.5, -16.0] + 1.0 * [9.0, 7.0, 7.0] = [3.5, -0.5, -9.0]
      //
      // weights for class 1.0:
      // current weights: [-3.5, -0.5, 11.0],
      // 5.0 * (0 - (1 / (1 + exp(-1 * [-3.5, -0.5, 11.0].[5.0, 7.0, 6.0]))))
      // 7.0 * (0 - (1 / (1 + exp(-1 * [-3.5, -0.5, 11.0].[5.0, 7.0, 6.0]))))
      // 6.0 * (0 - (1 / (1 + exp(-1 * [-3.5, -0.5, 11.0].[5.0, 7.0, 6.0]))))
      //-----------------------------------------------------------------
      // [-5.0, -7.0, -6.0]
      //
      // 1.0 * (1 - (1 / (1 + exp(-1 * [-3.5, -0.5, 11.0].[1.0, 2.0, 3.0]))))
      // 2.0 * (1 - (1 / (1 + exp(-1 * [-3.5, -0.5, 11.0].[1.0, 2.0, 3.0]))))
      // 3.0 * (1 - (1 / (1 + exp(-1 * [-3.5, -0.5, 11.0].[1.0, 2.0, 3.0]))))
      //-----------------------------------------------------------------
      // [4.19e-13, 8.38e-13, 1.25e-12]
      //
      // 10.0 * (1 - (1 / (1 + exp(-1 * [-3.5, -0.5, 11.0].[10.0, 12.0, 31.0]))))
      // 12.0 * (1 - (1 / (1 + exp(-1 * [-3.5, -0.5, 11.0].[10.0, 12.0, 31.0]))))
      // 31.0 * (1 - (1 / (1 + exp(-1 * [-3.5, -0.5, 11.0].[10.0, 12.0, 31.0]))))
      //-----------------------------------------------------------------
      // [0.0, 0.0, 0.0]
      //
      // 9.0 * (0 - (1 / (1 + exp(-1 * [-3.5, -0.5, 11.0].[9.0, 8.0, 5.0]))))
      // 8.0 * (0 - (1 / (1 + exp(-1 * [-3.5, -0.5, 11.0].[9.0, 8.0, 5.0]))))
      // 5.0 * (0 - (1 / (1 + exp(-1 * [-3.5, -0.5, 11.0].[9.0, 8.0, 5.0]))))
      //-----------------------------------------------------------------
      // [-8.9, -7.9, -4.9]
      //
      // 4.0 * (0 - (1 / (1 + exp(-1 * [-3.5, -0.5, 11.0].[4.0, 0.0, 1.0]))))
      // 0.0 * (0 - (1 / (1 + exp(-1 * [-3.5, -0.5, 11.0].[4.0, 0.0, 1.0]))))
      // 1.0 * (0 - (1 / (1 + exp(-1 * [-3.5, -0.5, 11.0].[4.0, 0.0, 1.0]))))
      //-----------------------------------------------------------------
      // [-0.189, 0.0, -0.047]
      //
      // derivative:
      // [-14.18, -14.89, -10.94]
      //
      // update:
      // [-3.5, -0.5, 11.0] + eta * [-14.18, -14.89, -10.94] = [-3.5, -0.5, 11.0] + 1.0 * [-14.18, -14.89, -10.94] =
      // = [-17.68, -15.4, -0.06]
      //
      // weights for class 2.0:
      // current weights: [-5.5, -6.5, -18.0]
      // 5.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -6.5, -18.0].[5.0, 7.0, 6.0])))) - dot sign means dot product
      // 7.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -6.5, -18.0].[5.0, 7.0, 6.0]))))
      // 6.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -6.5, -18.0].[5.0, 7.0, 6.0]))))
      //-----------------------------------------------------------------
      // [-1.23e-78, -1.72e-78, -1.48e-78]
      //
      // 1.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -6.5, -18.0].[1.0, 2.0, 3.0]))))
      // 2.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -6.5, -18.0].[1.0, 2.0, 3.0]))))
      // 3.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -6.5, -18.0].[1.0, 2.0, 3.0]))))
      //-----------------------------------------------------------------
      // [-5.38e-32, -1.07e-31, -1.61e-31]
      //
      // 10.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -6.5, -18.0].[10.0, 12.0, 31.0]))))
      // 12.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -6.5, -18.0].[10.0, 12.0, 31.0]))))
      // 31.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -6.5, -18.0].[10.0, 12.0, 31.0]))))
      //-----------------------------------------------------------------
      // [-7.98e-300, -9.58e-300, -2.47e-299]
      //
      // 9.0 * (1 - (1 / (1 + exp(-1 * [-5.5, -6.5, -18.0].[9.0, 8.0, 5.0]))))
      // 8.0 * (1 - (1 / (1 + exp(-1 * [-5.5, -6.5, -18.0].[9.0, 8.0, 5.0]))))
      // 5.0 * (1 - (1 / (1 + exp(-1 * [-5.5, -6.5, -18.0].[9.0, 8.0, 5.0]))))
      //-----------------------------------------------------------------
      // [9.0, 8.0, 5.0]
      //
      // 4.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -6.5, -18.0].[4.0, 0.0, 1.0]))))
      // 0.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -6.5, -18.0].[4.0, 0.0, 1.0]))))
      // 1.0 * (0 - (1 / (1 + exp(-1 * [-5.5, -6.5, -18.0].[4.0, 0.0, 1.0]))))
      //-----------------------------------------------------------------
      // [-1.69e-17, 0.0, -4.24e-18]
      //
      // derivative:
      // [9.0, 8.0, 5.0]
      //
      // update:
      // [-5.5, -6.5, -18.0] + eta * [9.0, 8.0, 5.0] = [-5.5, -6.5, -18.0] + 1.0 * [9.0, 8.0, 5.0] = [3.5, 1.5, -13.0]

      final weights = classifier.weightsByClasses.transpose();
      final class1Weights = weights.getRow(0).toList();
      final class2Weights = weights.getRow(1).toList();
      final class3Weights = weights.getRow(2).toList();

      expect(class1Weights, equals([3.5, -0.5, -9.0]));

      expect(class2Weights[0], closeTo(-17.68, 0.01));
      expect(class2Weights[1], closeTo(-15.4, 0.1));
      expect(class2Weights[2], closeTo(-0.06, 0.02));

      expect(class3Weights, equals([3.5, 1.5, -13.0]));
    });

    test('should make prediction', () {
      final features = MLMatrix.from([
        [5.0, 7.0, 6.0],
        [1.0, 2.0, 3.0],
        [10.0, 12.0, 31.0],
        [9.0, 8.0, 5.0],
        [4.0, 0.0, 1.0],
      ]);
      final labels = MLVector.from([0.0, 1.0, 1.0, 2.0, 0.0]);
      classifier.fit(features, labels);

      final newFeatures = MLMatrix.from([
        [2.0, 4.0, 1.0],
      ]);
      final probabilities = classifier.predictProbabilities(newFeatures);
      final classes = classifier.predictClasses(newFeatures);

      expect(
          probabilities,
          equals([
            [0.01798621006309986, 0.0, 0.5]
          ]));
      expect(classes, equals([2]));
    });

    test('should evaluate prediction quality, error = 1', () {
      final features = MLMatrix.from([
        [5.0, 7.0, 6.0],
        [1.0, 2.0, 3.0],
        [10.0, 12.0, 31.0],
        [9.0, 8.0, 5.0],
        [4.0, 0.0, 1.0],
      ]);
      final labels = MLVector.from([0.0, 1.0, 1.0, 2.0, 0.0]);
      classifier.fit(features, labels);

      final newFeatures = MLMatrix.from([
        [2.0, 4.0, 1.0],
      ]);
      final origLabels = MLVector.from([1.0]);
      final error = classifier.test(newFeatures, origLabels, MetricType.accuracy);
      expect(error, equals(1.0));
    });

    test('should evaluate prediction quality, error = 0', () {
      final features = MLMatrix.from([
        [5.0, 7.0, 6.0],
        [1.0, 2.0, 3.0],
        [10.0, 12.0, 31.0],
        [9.0, 8.0, 5.0],
        [4.0, 0.0, 1.0],
      ]);
      final labels = MLVector.from([0.0, 1.0, 1.0, 2.0, 0.0]);
      classifier.fit(features, labels);

      final newFeatures = MLMatrix.from([
        [2.0, 4.0, 1.0],
      ]);
      final origLabels = MLVector.from([2.0]);
      final error = classifier.test(newFeatures, origLabels, MetricType.accuracy);
      expect(error, equals(0.0));
    });

    test('should consider intercept term', () {
      final classifier = LogisticRegressor(
          batchSize: 2,
          iterationLimit: 1,
          learningRateType: LearningRateType.constant,
          learningRate: 1.0,
          fitIntercept: true);
      final features = MLMatrix.from([
        [5.0, 7.0, 6.0],
        [1.0, 2.0, 3.0],
      ]);
      final labels = MLVector.from([0.0, 1.0]);
      classifier.fit(features, labels);

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
      expect(
          classifier.weightsByClasses.transpose(),
          equals([
            [0.0, 2.0, 2.5, 1.5],
            [0.0, -2.0, -2.5, -1.5]
          ]));
    });

    test('should consider intercept scale if intercept term is going to be fitted', () {
      final classifier = LogisticRegressor(
          batchSize: 3,
          iterationLimit: 1,
          learningRateType: LearningRateType.constant,
          learningRate: 1.0,
          fitIntercept: true,
          interceptScale: 2.0);
      final features = MLMatrix.from([
        [5.0, 7.0, 6.0],
        [1.0, 2.0, 3.0],
        [3.0, 4.0, 5.0],
      ]);
      final labels = MLVector.from([0.0, 1.0, 0.0]);
      classifier.fit(features, labels);

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
      expect(
          classifier.weightsByClasses.transpose(),
          equals([
            [1.0, 3.5, 4.5, 4.0],
            [-1.0, -3.5, -4.5, -4.0]
          ]));
    });
  });
}
