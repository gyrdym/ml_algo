import 'package:ml_algo/src/optimizer/convergence_detector/convergence_detector.dart';
import 'package:ml_linalg/vector.dart';

class ConvergenceDetectorImpl implements ConvergenceDetector {
  @override
  final double minUpdate;

  @override
  final int iterationsLimit;

  ConvergenceDetectorImpl(this.minUpdate, this.iterationsLimit);

  @override
  bool isConverged(MLVector coefficientUpdates, int iteration) {
    if (iterationsLimit != null && iteration >= iterationsLimit) {
      return true;
    }

    if (coefficientUpdates.min() <= minUpdate) {
      return true;
    }

    return false;
  }
}
