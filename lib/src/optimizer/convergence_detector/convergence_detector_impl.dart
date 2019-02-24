import 'package:ml_algo/src/optimizer/convergence_detector/convergence_detector.dart';

class ConvergenceDetectorImpl implements ConvergenceDetector {
  ConvergenceDetectorImpl(this.minDiff, this.iterationsLimit) {
    if (minDiff == null && iterationsLimit == null) {
      throw Exception();
    }
  }

  @override
  final double minDiff;

  @override
  final int iterationsLimit;

  @override
  bool isConverged(double coefficientsDiff, int iteration) {
    if (iterationsLimit != null && iteration >= iterationsLimit) {
      return true;
    }

    if (minDiff != null && coefficientsDiff <= minDiff) {
      return true;
    }

    return false;
  }
}
