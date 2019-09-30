import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector_impl.dart';
import 'package:test/test.dart';

void main() {
  group('ConvergenceDetectorImpl', () {
    test('should not detect convergence if passed iteration number is less '
        'than iterations limit and minimal passed update is greater than the '
        'minimum update', () {
      final iterationsLimit = 100;
      final minUpdate = 0.5;
      final update = 3.0;
      final detector = ConvergenceDetectorImpl(minUpdate, iterationsLimit);
      expect(detector.isConverged(update, 99), isFalse);
    });

    test('should detect convergence if passed iteration number is greater than '
        'iterations limit and minimal passed update is greater than the '
        'minimum update', () {
      final iterationsLimit = 100;
      final minUpdate = 0.5;
      final update = 1.0;
      final detector = ConvergenceDetectorImpl(minUpdate, iterationsLimit);
      expect(detector.isConverged(update, 101), isTrue);
    });

    test('should detect convergence if passed iteration number is equal to '
        'iterations limit and minimal passed update is greater than the '
        'minimum update', () {
      final iterationsLimit = 100;
      final minUpdate = 0.5;
      final update = 2.0;
      final detector = ConvergenceDetectorImpl(minUpdate, iterationsLimit);
      expect(detector.isConverged(update, 100), isTrue);
    });

    test('should detect convergence if minimal passed update is less than the '
        'minimum update', () {
      final iterationsLimit = 100;
      final minUpdate = 0.5;
      final update = 0.3;
      final detector = ConvergenceDetectorImpl(minUpdate, iterationsLimit);
      expect(detector.isConverged(update, 10), isTrue);
    });

    test('should not throw error if the iteration limit is null', () {
      final minUpdate = 0.5;
      final update = 0.3;
      final detector = ConvergenceDetectorImpl(minUpdate, null);
      expect(detector.isConverged(update, 10), isTrue);
    });
  });
}
