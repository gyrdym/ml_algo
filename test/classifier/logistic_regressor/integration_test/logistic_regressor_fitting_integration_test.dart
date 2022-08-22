import 'package:ml_algo/src/classifier/logistic_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/common/exception/unsupported_linear_optimizer_type_exception.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:test/test.dart';

import '../../../helpers.dart';

void main() {
  group('LogisticRegressor', () {
    final createClassifier = ({
      required DataFrame samples,
      required String targetName,
      int iterationsLimit = 1,
      LearningRateType learningRateType = LearningRateType.constant,
      double initialLearningRate = 1.0,
      int batchSize = 3,
      bool fitIntercept = true,
      double interceptScale = 2.0,
      bool collectLearningData = false,
      required DType dtype,
    }) =>
        LogisticRegressor(
          samples,
          targetName,
          optimizerType: LinearOptimizerType.gradient,
          iterationsLimit: iterationsLimit,
          learningRateType: learningRateType,
          initialLearningRate: initialLearningRate,
          batchSize: batchSize,
          fitIntercept: fitIntercept,
          interceptScale: interceptScale,
          collectLearningData: collectLearningData,
          dtype: dtype,
        );

    group('default constructor (fitIntercept=false)', () {
      final data = <Iterable<num>>[
        [5.0, 7.0, 6.0, 1.0],
        [1.0, 2.0, 3.0, 0.0],
        [10.0, 12.0, 31.0, 0.0],
        [9.0, 8.0, 5.0, 0.0],
        [4.0, 0.0, 1.0, 1.0],
      ];
      final targetName = 'col_3';
      final samples = DataFrame(data, headerExists: false);

      tearDown(() {
        injector.clearAll();
        logisticRegressorInjector.clearAll();
      });

      test('should fit given data, float32 case', () {
        final classifier = LogisticRegressor(
          samples,
          targetName,
          optimizerType: LinearOptimizerType.gradient,
          iterationsLimit: 2,
          learningRateType: LearningRateType.constant,
          initialLearningRate: 1.0,
          batchSize: 5,
          fitIntercept: false,
        );

        expect(
            classifier.coefficientsByClasses,
            iterable2dAlmostEqualTo([
              [
                3.5,
              ],
              [
                -0.5,
              ],
              [
                -9.0,
              ],
            ], 1e-2));
      });

      test('should fit given data, float64 case', () {
        final classifier = LogisticRegressor(
          samples,
          targetName,
          optimizerType: LinearOptimizerType.gradient,
          iterationsLimit: 2,
          learningRateType: LearningRateType.constant,
          initialLearningRate: 1.0,
          batchSize: 5,
          fitIntercept: false,
          dtype: DType.float64,
        );

        expect(
            classifier.coefficientsByClasses,
            iterable2dAlmostEqualTo([
              [
                3.5,
              ],
              [
                -0.5,
              ],
              [
                -9.0,
              ],
            ], 1e-2));
      });
    });

    group('default constructor (fitIntercept=true)', () {
      final features = DataFrame(<Iterable<num>>[
        [5.0, 7.0, 6.0, 1.0],
        [1.0, 2.0, 3.0, 0.0],
      ], headerExists: false);
      final targetName = 'col_3';

      tearDown(() {
        injector.clearAll();
        logisticRegressorInjector.clearAll();
      });

      test('should consider intercept term, dtype=DType.float32', () {
        final classifier = createClassifier(
            samples: features,
            targetName: targetName,
            dtype: DType.float32,
            batchSize: 2);

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
            classifier.coefficientsByClasses,
            equals([
              [0.0],
              [2.0],
              [2.5],
              [1.5],
            ]));
      });

      test('should consider intercept term, dtype=DType.float64', () {
        final classifier = createClassifier(
            samples: features,
            targetName: targetName,
            batchSize: 2,
            dtype: DType.float64);

        expect(
            classifier.coefficientsByClasses,
            equals([
              [0.0],
              [2.0],
              [2.5],
              [1.5],
            ]));
      });
    });

    group('default constructor (interceptScale=2.0)', () {
      final samples = DataFrame(<Iterable<num>>[
        [5.0, 7.0, 6.0, 1.0],
        [1.0, 2.0, 3.0, 0.0],
        [3.0, 4.0, 5.0, 1.0],
      ], headerExists: false);
      final targetName = 'col_3';

      test(
          'should consider intercept scale if intercept term is going to be '
          'fitted, dtype=DType.float32', () {
        final classifier = createClassifier(
            samples: samples, targetName: targetName, dtype: DType.float32);

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
        expect(
            classifier.coefficientsByClasses,
            equals([
              [1.0],
              [3.5],
              [4.5],
              [4.0],
            ]));
      });

      test(
          'should consider intercept scale if intercept term is going to be '
          'fitted, dtype=DType.float64', () {
        final classifier = createClassifier(
            samples: samples, targetName: targetName, dtype: DType.float64);

        expect(
            classifier.coefficientsByClasses,
            equals([
              [1.0],
              [3.5],
              [4.5],
              [4.0],
            ]));
      });
    });

    group('default constructor (collectLearningData=true)', () {
      final samples = DataFrame(<Iterable<num>>[
        [5.0, 7.0, 6.0, 1.0],
        [1.0, 2.0, 3.0, 0.0],
        [3.0, 4.0, 5.0, 1.0],
      ], headerExists: false);
      final targetName = 'col_3';

      test('should return cost per iteration list', () {
        final classifier = LogisticRegressor(samples, targetName,
            optimizerType: LinearOptimizerType.gradient,
            batchSize: 3,
            dtype: DType.float32,
            collectLearningData: true,
            fitIntercept: false,
            iterationsLimit: 3);

        expect(classifier.costPerIteration, [
          closeTo(-2.0794, 1e-4),
          closeTo(-2.0319, 1e-4),
          closeTo(-1.9884, 1e-4),
        ]);
      });
    });

    group('default constructor (optimizerType=LinearOptimizerType.coordinate)',
        () {
      final samples = DataFrame(<Iterable<dynamic>>[
        ['col_1', 'col_2'],
        [1, 2]
      ]);

      test('should throw exception', () {
        final createClassifier = () => LogisticRegressor(samples, 'col_1',
            optimizerType: LinearOptimizerType.coordinate);

        expect(createClassifier,
            throwsA(isA<UnsupportedLinearOptimizerTypeException>()));
      });
    });

    group('default constructor (optimizerType=LinearOptimizerType.closedForm)',
        () {
      final samples = DataFrame(<Iterable<dynamic>>[
        ['col_1', 'col_2'],
        [1, 2]
      ]);

      test('should throw exception', () {
        final createClassifier = () => LogisticRegressor(samples, 'col_1',
            optimizerType: LinearOptimizerType.closedForm);

        expect(createClassifier,
            throwsA(isA<UnsupportedLinearOptimizerTypeException>()));
      });
    });
  });
}
