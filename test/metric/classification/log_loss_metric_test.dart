import 'package:ml_algo/src/metric/classification/log_loss_metric.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

void main() {
  group('LogLossMetric', () {
    const metric = LogLossMetric();

    test('perfect predictions → loss ≈ 0', () {
      final yTrue = Matrix.column([1, 0, 1, 0]);
      final yPred = Matrix.column([1.0, 0.0, 1.0, 0.0]);
      expect(metric.getScore(yPred, yTrue), closeTo(0.0, 1e-12));
    });

    test('typical predictions', () {
      final yTrue = Matrix.column([1, 0]);
      final yPred = Matrix.column([0.9, 0.1]);
      expect(metric.getScore(yPred, yTrue),
          closeTo(0.10536051565782628, 1e-6)); // -ln(0.9)
    });

    test('probabilities are clipped', () {
      final yTrue = Matrix.column([1, 0]);
      final yPred = Matrix.column([0.0, 1.0]);
      expect(metric.getScore(yPred, yTrue).isFinite, isTrue);
    });
  });
}
