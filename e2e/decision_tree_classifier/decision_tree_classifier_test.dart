import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:test/test.dart';

Future<Vector> evaluateClassifier(MetricType metric, DType dtype) async {
  final samples = (await fromCsv('e2e/_datasets/iris.csv'))
      .shuffle()
      .dropSeries(seriesNames: ['Id']);
  final pipeline = Pipeline(samples, [
    encodeAsIntegerLabels(
      featureNames: ['Species'],
    ),
  ]);
  final numberOfFolds = 5;
  final processed = pipeline.process(samples);
  final validator = CrossValidator.kFold(
    processed,
    numberOfFolds: numberOfFolds,
  );
  final createClassifier = (DataFrame trainSamples) =>
      DecisionTreeClassifier(
        trainSamples,
        'Species',
        minError: 0.3,
        minSamplesCount: 5,
        maxDepth: 4,
        dtype: dtype,
      );

  return validator.evaluate(
    createClassifier,
    metric,
  );
}

void main() async {
  group('DecisionTreeClassifier', () {
    test('should return adequate score on iris dataset using accuracy '
        'metric, dtype=DType.float32', () async {
      final scores = await evaluateClassifier(
          MetricType.accuracy, DType.float32);

      expect(scores.mean(), greaterThan(0.5));
    });

    test('should return adequate score on iris dataset using accuracy '
        'metric, dtype=DType.float64', () async {
      final scores = await evaluateClassifier(
          MetricType.accuracy, DType.float64);

      expect(scores.mean(), greaterThan(0.5));
    });

    test('should return adequate score on iris dataset using precision '
        'metric, dtype=DType.float32', () async {
      final scores = await evaluateClassifier(
          MetricType.precision, DType.float32);

      expect(scores.mean(), greaterThan(0.5));
    });

    test('should return adequate score on iris dataset using precision '
        'metric, dtype=DType.float64', () async {
      final scores = await evaluateClassifier(
          MetricType.precision, DType.float64);

      expect(scores.mean(), greaterThan(0.5));
    });

    test('should return adequate score on iris dataset using recall '
        'metric, dtype=DType.float32', () async {
      final scores = await evaluateClassifier(
          MetricType.recall, DType.float32);

      expect(scores.mean(), greaterThan(0.5));
    });

    test('should return adequate score on iris dataset using recall '
        'metric, dtype=DType.float64', () async {
      final scores = await evaluateClassifier(
          MetricType.recall, DType.float64);

      expect(scores.mean(), greaterThan(0.5));
    });
  });
}
