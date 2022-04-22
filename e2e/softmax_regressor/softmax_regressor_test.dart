import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:test/test.dart';

Future<Vector> evaluateSoftmaxRegressor(
    MetricType metricType, DType dtype) async {
  final samples = (await fromCsv('e2e/_datasets/iris.csv'))
      .shuffle()
      .dropSeries(names: ['Id']);
  final pipeline = Pipeline(samples, [
    toOneHotLabels(
      columnNames: ['Species'],
    ),
  ]);
  final classNames = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'];
  final processed = pipeline.process(samples);
  final numberOfFolds = 5;
  final validator = CrossValidator.kFold(
    processed,
    numberOfFolds: numberOfFolds,
  );
  final predictorFactory = (DataFrame trainingSamples) => SoftmaxRegressor(
        trainingSamples,
        classNames,
        initialLearningRate: 0.035,
        iterationsLimit: 5000,
        minCoefficientsUpdate: 1e-1000000,
        learningRateType: LearningRateType.constant,
        dtype: dtype,
      );

  return validator.evaluate(
    predictorFactory,
    metricType,
  );
}

Future main() async {
  group('SoftmaxRegressor', () {
    test(
        'should return adequate score on iris dataset using accuracy '
        'metric, dtype=DType.float32', () async {
      final scores =
          await evaluateSoftmaxRegressor(MetricType.accuracy, DType.float32);

      expect(scores.mean(), greaterThan(0.5));
    });

    test(
        'should return adequate score on iris dataset using accuracy '
        'metric, dtype=DType.float64', () async {
      final scores =
          await evaluateSoftmaxRegressor(MetricType.accuracy, DType.float64);

      expect(scores.mean(), greaterThan(0.5));
    });

    test(
        'should return adequate score on iris dataset using precision '
        'metric, dtype=DType.float32', () async {
      final scores =
          await evaluateSoftmaxRegressor(MetricType.precision, DType.float32);

      expect(scores.mean(), greaterThan(0.5));
    });

    test(
        'should return adequate score on iris dataset using precision '
        'metric, dtype=DType.float64', () async {
      final scores =
          await evaluateSoftmaxRegressor(MetricType.precision, DType.float64);

      expect(scores.mean(), greaterThan(0.5));
    });

    test(
        'should return adequate score on iris dataset using recall '
        'metric, dtype=DType.float32', () async {
      final scores =
          await evaluateSoftmaxRegressor(MetricType.recall, DType.float32);

      expect(scores.mean(), greaterThan(0.5));
    });

    test(
        'should return adequate score on iris dataset using recall '
        'metric, dtype=DType.float64', () async {
      final scores =
          await evaluateSoftmaxRegressor(MetricType.recall, DType.float64);

      expect(scores.mean(), greaterThan(0.5));
    });
  });
}
