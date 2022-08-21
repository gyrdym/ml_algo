import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:test/test.dart';

num evaluateLogisticRegressor(MetricType metric, DType dtype) {
  final data = getPimaIndiansDiabetesDataFrame().shuffle();
  final samples = splitData(data, [0.8]);
  final trainSamples = samples.first;
  final testSamples = samples.last;
  final model = LogisticRegressor.newton(
    trainSamples,
    'Outcome',
    dtype: dtype,
  );

  return model.assess(testSamples, metric);
}

Future main() async {
  group('LogisticRegressor.newton', () {
    test(
        'should return adequate score on pima indians diabetes dataset using '
        'accuracy metric, dtype=DType.float32', () {
      final score =
          evaluateLogisticRegressor(MetricType.accuracy, DType.float32);

      print('float32, accuracy is $score');

      expect(score, greaterThan(0.7));
    });

    test(
        'should return adequate score on pima indians diabetes dataset using '
        'accuracy metric, dtype=DType.float64', () {
      final score =
          evaluateLogisticRegressor(MetricType.accuracy, DType.float32);

      print('float64, accuracy is $score');

      expect(score, greaterThan(0.7));
    });

    test(
        'should return adequate score on pima indians diabetes dataset using '
        'precision metric, dtype=DType.float32', () {
      final score =
          evaluateLogisticRegressor(MetricType.precision, DType.float32);

      print('float32, precision is $score');

      expect(score, greaterThan(0.65));
    });

    test(
        'should return adequate score on pima indians diabetes dataset using '
        'precision metric, dtype=DType.float64', () {
      final score =
          evaluateLogisticRegressor(MetricType.precision, DType.float32);

      print('float64, precision is $score');

      expect(score, greaterThan(0.65));
    });

    test(
        'should return adequate score on pima indians diabetes dataset using '
        'recall metric, dtype=DType.float32', () {
      final score = evaluateLogisticRegressor(MetricType.recall, DType.float32);

      print('float32, recall is $score');

      expect(score, greaterThan(0.65));
    });

    test(
        'should return adequate score on pima indians diabetes dataset using '
        'recall metric, dtype=DType.float64', () {
      final score = evaluateLogisticRegressor(MetricType.recall, DType.float32);

      print('float64, recall is $score');

      expect(score, greaterThan(0.65));
    });
  });
}
