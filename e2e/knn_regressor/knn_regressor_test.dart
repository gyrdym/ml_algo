import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/vector.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:test/test.dart';

Future<Vector> evaluateKnnRegressor(MetricType metricType, DType dtype) async {
  final data = (await fromCsv(
    'e2e/_datasets/housing.csv',
    headerExists: false,
    columnDelimiter: ' ',
  ))
      .shuffle();
  final normalized = Normalizer().process(data);
  final folds = 5;
  final targetName = 'col_13';
  final validator = CrossValidator.kFold(
    normalized,
    numberOfFolds: folds,
    dtype: dtype,
  );

  return validator.evaluate(
      (trainSamples) => KnnRegressor(trainSamples, targetName, folds),
      metricType);
}

void main() {
  group('KnnRegressor', () {
    test(
        'should return adequate score on boston housing dataset using mape '
        'metric, dtype=DType.float32', () async {
      final scores = await evaluateKnnRegressor(MetricType.mape, DType.float32);

      expect(scores.mean(), lessThan(50));
    });

    test(
        'should return adequate score on boston housing dataset using mape '
        'metric, dtype=DType.float64', () async {
      final scores = await evaluateKnnRegressor(MetricType.mape, DType.float64);

      expect(scores.mean(), lessThan(0.5));
    });
  });
}
