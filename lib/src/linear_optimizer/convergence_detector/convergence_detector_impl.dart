import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector.dart';

class ConvergenceDetectorImpl implements ConvergenceDetector {
  ConvergenceDetectorImpl(this.minDiff, this.iterationsLimit);

  @override
  final double minDiff;

  @override
  final int iterationsLimit;

  @override
  bool isConverged(double coefficientsDiff, int iteration) {
    return iteration >= iterationsLimit
        || coefficientsDiff <= minDiff;
  }
}
