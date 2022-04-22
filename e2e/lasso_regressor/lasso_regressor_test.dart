import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

Future<Vector> evaluateLassoRegressor(
    MetricType metricType, DType dtype) async {
  final samples = (await fromCsv('e2e/_datasets/advertising.csv'))
      .shuffle()
      .dropSeries(names: ['Num']);
  final targetName = 'Sales';
  final validator = CrossValidator.kFold(
    samples,
    numberOfFolds: 5,
  );

  return validator.evaluate(
      (trainSamples) => LinearRegressor.lasso(
            trainSamples,
            targetName,
            iterationLimit: 100,
            lambda: 50000.0,
            dtype: dtype,
          ),
      metricType);
}

void main() {
  group('LassoRegressor', () {
    test(
        'should return adequate error on mape metric, '
        'dtype=DType.float32', () async {
      final scores =
          await evaluateLassoRegressor(MetricType.mape, DType.float32);

      expect(scores.mean(), lessThan(0.5));
    });

    test(
        'should return adequate error on mape metric, '
        'dtype=DType.float64', () async {
      final scores =
          await evaluateLassoRegressor(MetricType.mape, DType.float64);

      expect(scores.mean(), lessThan(0.5));
    });
  });
}
