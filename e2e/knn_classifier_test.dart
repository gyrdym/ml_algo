import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:test/test.dart';

Future<Vector> evaluateKnnClassifier(MetricType metric, DType dtype) async {
  final samples = (await fromCsv('e2e/datasets/iris.csv'))
      .shuffle()
      .dropSeries(seriesNames: ['Id']);
  final targetName = 'Species';
  final pipeline = Pipeline(samples, [
    encodeAsIntegerLabels(
      featureNames: [targetName],
    ),
  ]);
  final processed = pipeline.process(samples);
  final numberOfFolds = 7;
  final numberOfNeighbours = 5;
  final validator = CrossValidator.kFold(
    processed,
    numberOfFolds: numberOfFolds,
  );
  final createClassifier = (DataFrame trainSamples) =>
      KnnClassifier(
        trainSamples,
        targetName,
        numberOfNeighbours,
        dtype: dtype,
      );

  return validator.evaluate(
    createClassifier,
    metric,
  );
}

void main() async {
  group('KnnClassifier', () {
    test('should return adequate score on iris dataset using accuracy '
        'metric, dtype=DType.float32', () async {
      final scores = await evaluateKnnClassifier(MetricType.accuracy,
          DType.float32);

      expect(scores.mean(), closeTo(0.95, 3e-2));
    });

    test('should return adequate score on iris dataset using accuracy '
        'metric, dtype=DType.float64', () async {
      final scores = await evaluateKnnClassifier(MetricType.accuracy,
          DType.float64);

      expect(scores.mean(), closeTo(0.95, 4e-2));
    });

    test('should return adequate score on iris dataset using precision '
        'metric, dtype=DType.float32', () async {
      final scores = await evaluateKnnClassifier(MetricType.precision,
          DType.float32);

      expect(scores.mean(), closeTo(0.95, 4e-2));
    });

    test('should return adequate score on iris dataset using precision '
        'metric, dtype=DType.float64', () async {
      final scores = await evaluateKnnClassifier(MetricType.precision,
          DType.float64);

      expect(scores.mean(), closeTo(0.65, 5e-2));
    });
  });
}
