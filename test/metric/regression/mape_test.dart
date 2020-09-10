import 'package:ml_algo/src/common/exception/matrix_column_exception.dart';
import 'package:ml_algo/src/metric/regression/mape.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

void main() {
  group('MAPE metric', () {
    const metric = MapeMetric();
    final predictedLabels = Matrix.column([12, 18, 12, 90, 78]);
    final originalLabels = Matrix.column([10, 20 , 30, 60, 70]);

    test('should throw an error if predicted labels matrix\'s columns count '
        'is empty', () {
      final actual = () => metric.getScore(Matrix.empty(), originalLabels);

      expect(actual, throwsA(isA<MatrixColumnException>()));
    });

    test('should throw an error if predicted labels matrix\'s columns count '
        'is greater than 1', () {
      final actual = () => metric.getScore(Matrix.row([1, 2]), originalLabels);

      expect(actual, throwsA(isA<MatrixColumnException>()));
    });

    test('should throw an error if original labels matrix\'s columns count '
        'is empty', () {
      final actual = () => metric.getScore(predictedLabels, Matrix.empty());

      expect(actual, throwsA(isA<MatrixColumnException>()));
    });

    test('should throw an error if original labels matrix\'s columns count '
        'is greater than 1', () {
      final actual = () => metric.getScore(predictedLabels, Matrix.row([1, 2]));

      expect(actual, throwsA(isA<MatrixColumnException>()));
    });

    test('should count score', () {
      final actual = metric.getScore(predictedLabels, originalLabels);

      expect(actual, closeTo(0.0628, 1e-4));
    });
  });
}
