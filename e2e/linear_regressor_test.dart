import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';

Future<Vector> evaluateLinearRegressor(MetricType metricType,
    DType dtype) async {
  final samples = (await fromCsv('e2e/datasets/housing.csv',
      headerExists: false,
      columnDelimiter: ' ')).shuffle();
  final folds = 5;
  final targetName = 'col_13';
  final validator = CrossValidator.kFold(
    samples,
    numberOfFolds: folds,
  );
  final createRegressor = (DataFrame trainSamples) =>
      LinearRegressor(
        trainSamples,
        targetName,
        optimizerType: LinearOptimizerType.gradient,
        initialLearningRate: 0.00000385,
        randomSeed: 2,
        learningRateType: LearningRateType.decreasingAdaptive,
        dtype: dtype,
      );

  return validator.evaluate(createRegressor, metricType);
}

void main() async {
  group('LinearRegressor', () {
    test('should return adequate error on mape metric, '
        'dtype=DType.float32', () async {
      expect(
        (await evaluateLinearRegressor(MetricType.mape, DType.float32)).mean(),
        lessThan(50),
      );
    });

    test('should return adequate error on mape metric, '
        'dtype=DType.float64', () async {
      expect(
        (await evaluateLinearRegressor(MetricType.mape, DType.float64)).mean(),
        lessThan(50),
      );
    });
  });
}
