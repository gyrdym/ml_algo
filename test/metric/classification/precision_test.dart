import 'package:ml_algo/src/metric/classification/precision.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('PrecisionMetric', () {
    final origLabels = Matrix.fromList([
      [1, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [0, 1, 0],
      [1, 0, 0],
      [0, 0, 1],
    ]);
    final origLabelsWithZeroColumn = Matrix.fromList([
      [1, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [1, 0, 0],
      [0, 1, 0],
      [1, 0, 0],
      [1, 0, 0],
    ]);
    final predictedLabels = Matrix.fromList([
      [0, 1, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [0, 1, 0],
      [1, 0, 0],
      [0, 0, 1],
    ]);
    final predictedLabelsWithZeroColumn = Matrix.fromList([
      [0, 1, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 1, 0],
      [0, 1, 0],
      [1, 0, 0],
      [1, 0, 0],
    ]);
    final metric = const PrecisionMetric();

    test('should return a correct score', () {
      final score = metric.getScore(predictedLabels, origLabels);

      expect(score, closeTo((1 + 2 / 3 + 1) / 3, 1e-5));
    });

    test(
        'should return a correct score if there is at least one column with '
        'all zeroes ', () {
      final score = metric.getScore(predictedLabelsWithZeroColumn, origLabels);

      expect(score, closeTo((2 / 3 + 2 / 4 + 0) / 3, 1e-5));
    });

    test(
        'should return a correct score if there is a zero column in the '
        'original labels', () {
      final score = metric.getScore(predictedLabels, origLabelsWithZeroColumn);

      expect(score, closeTo((1 + 2 / 3 + 0) / 3, 1e-5));
    });

    test(
        'should return a correct score if both original labels and predicted '
        'labels have zero columns', () {
      final score = metric.getScore(
          predictedLabelsWithZeroColumn, origLabelsWithZeroColumn);

      expect(score, closeTo((1 + 2 / 4 + 0) / 3, 1e-5));
    });

    test('should return 1 if predicted labels are correct', () {
      final score = metric.getScore(origLabels, origLabels);

      expect(score, 1);
    });
  });
}
