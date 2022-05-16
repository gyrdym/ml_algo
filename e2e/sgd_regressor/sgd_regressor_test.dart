import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

Future<Vector> evaluateSgdRegressor(MetricType metricType, DType dtype) async {
  final samples = (await fromCsv('e2e/_datasets/housing.csv',
          headerExists: false, columnDelimiter: ' '))
      .shuffle();
  final folds = 5;
  final targetName = 'col_13';
  final validator = CrossValidator.kFold(
    samples,
    numberOfFolds: folds,
  );
  final createRegressor = (DataFrame trainSamples) => LinearRegressor.SGD(
        trainSamples,
        targetName,
        initialLearningRate: 1e-6,
        learningRateType: LearningRateType.stepBased,
        dropRate: 3,
        dtype: dtype,
      );

  return validator.evaluate(createRegressor, metricType);
}

void main() async {
  group('SGDRegressor', () {
    test(
        'should return adequate error on mape metric, '
        'dtype=DType.float32', () async {
      expect(
        (await evaluateSgdRegressor(MetricType.mape, DType.float32)).mean(),
        lessThan(0.5),
      );
    });

    test(
        'should return adequate error on mape metric, '
        'dtype=DType.float64', () async {
      expect(
        (await evaluateSgdRegressor(MetricType.mape, DType.float64)).mean(),
        lessThan(0.5),
      );
    });
  });
}
