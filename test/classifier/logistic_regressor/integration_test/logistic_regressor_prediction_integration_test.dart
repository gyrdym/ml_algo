import 'package:ml_algo/src/classifier/logistic_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

void main() {
  final testPredictMethod = (DType dtype) {
    final data = <Iterable<num>>[
      [5.0, 7.0, 6.0, 1.0],
      [1.0, 2.0, 3.0, 0.0],
      [10.0, 12.0, 31.0, 0.0],
      [9.0, 8.0, 5.0, 0.0],
      [4.0, 0.0, 1.0, 1.0],
    ];
    final targetName = 'col_3';
    final samples = DataFrame(data, headerExists: false);
    late LogisticRegressor classifier;

    setUp(() {
      injector.clearAll();
      logisticRegressorInjector.clearAll();

      classifier = LogisticRegressor(
        samples,
        targetName,
        iterationsLimit: 2,
        learningRateType: LearningRateType.constant,
        initialLearningRate: 1.0,
        batchSize: 5,
        fitIntercept: false,
        dtype: dtype,
      );
    });

    test('should make prediction', () {
      final newFeatures = Matrix.fromList([
        [2.0, 4.0, 1.0],
      ], dtype: dtype);

      final probabilities = classifier.predictProbabilities(
        DataFrame.fromMatrix(newFeatures),
      );

      final classes = classifier.predict(
        DataFrame.fromMatrix(newFeatures),
      );

      expect(probabilities.header, equals(['col_3']));
      expect(
          probabilities.toMatrix(),
          equals([
            [0.01798621006309986]
          ]));
      expect(classes.header, equals(['col_3']));
      expect(
          classes.toMatrix(),
          equals([
            [0.0]
          ]));
    });

    test('should evaluate prediction quality, accuracy = 0', () {
      final newSamples = DataFrame([
        <num>[2.0, 4.0, 1.0, 1.0],
      ], headerExists: false);

      final score = classifier.assess(newSamples, MetricType.accuracy);

      expect(score, equals(0.0));
    });

    test('should evaluate prediction quality, accuracy = 1', () {
      final newFeatures = DataFrame([
        <num>[2, 4, 1, 0],
      ], headerExists: false);

      final score = classifier.assess(newFeatures, MetricType.accuracy);

      expect(score, equals(1.0));
    });
  };

  group('LogisticRegressor.predict', () {
    group('(dtype=DType.float32)', () {
      testPredictMethod(DType.float32);
    });

    group('(dtype=DType.float64)', () {
      testPredictMethod(DType.float64);
    });
  });
}
