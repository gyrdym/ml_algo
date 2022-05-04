import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

Future<Vector> evaluateLogisticRegressor(MetricType metric, DType dtype) {
  final samples = getPimaIndiansDiabetesDataFrame().shuffle();
  final numberOfFolds = 5;
  final targetNames = ['Outcome'];
  final validator = CrossValidator.kFold(
    samples,
    numberOfFolds: numberOfFolds,
  );
  final createClassifier = (DataFrame trainSamples) => LogisticRegressor(
        trainSamples,
        targetNames.first,
        optimizerType: LinearOptimizerType.gradient,
        iterationsLimit: 100,
        learningRateType: LearningRateType.exponential,
        batchSize: trainSamples.rows.length,
        probabilityThreshold: 0.5,
        dtype: dtype,
      );

  return validator.evaluate(
    createClassifier,
    metric,
  );
}

Future main() async {
  group('LogisticRegressor', () {
    test(
        'should return adequate score on pima indians diabetes dataset using '
        'accuracy metric, dtype=DType.float32', () async {
      final scores =
          await evaluateLogisticRegressor(MetricType.accuracy, DType.float32);

      expect(scores.mean(), greaterThan(0.5));
    });

    test(
        'should return adequate score on pima indians diabetes dataset using '
        'accuracy metric, dtype=DType.float64', () async {
      final scores =
          await evaluateLogisticRegressor(MetricType.accuracy, DType.float32);

      expect(scores.mean(), greaterThan(0.5));
    });

    test(
        'should return adequate score on pima indians diabetes dataset using '
        'precision metric, dtype=DType.float32', () async {
      final scores =
          await evaluateLogisticRegressor(MetricType.precision, DType.float32);

      expect(scores.mean(), greaterThan(0.5));
    });

    test(
        'should return adequate score on pima indians diabetes dataset using '
        'precision metric, dtype=DType.float64', () async {
      final scores =
          await evaluateLogisticRegressor(MetricType.precision, DType.float32);

      expect(scores.mean(), greaterThan(0.5));
    });

    test(
        'should return adequate score on pima indians diabetes dataset using '
        'recall metric, dtype=DType.float32', () async {
      final scores =
          await evaluateLogisticRegressor(MetricType.recall, DType.float32);

      expect(scores.mean(), greaterThan(0.5));
    });

    test(
        'should return adequate score on pima indians diabetes dataset using '
        'recall metric, dtype=DType.float64', () async {
      final scores =
          await evaluateLogisticRegressor(MetricType.recall, DType.float32);

      expect(scores.mean(), greaterThan(0.5));
    });
  });
}
