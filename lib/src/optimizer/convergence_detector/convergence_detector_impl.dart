import 'package:ml_algo/src/optimizer/convergence_detector/convergence_detector.dart';

class ConvergenceDetectorImpl implements ConvergenceDetector {
  @override
  final double minDiff;

  @override
  final int iterationsLimit;

  ConvergenceDetectorImpl(this.minDiff, this.iterationsLimit);

  @override
  bool isConverged(double coefficientsDiff, int iteration) {
    if (iterationsLimit != null && iteration >= iterationsLimit) {
      return true;
    }

    if (coefficientsDiff <= minDiff) {
      return true;
    }

    return false;
  }
}
