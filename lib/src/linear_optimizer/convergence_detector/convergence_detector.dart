///
/// An entity that decides, when to stop an optimization algorithm
abstract class ConvergenceDetector {
  double get minDiff;
  int get iterationsLimit;

  /// The algorithm considered converged if it hits one of the limits:
  ///
  ///  - if the passed iteration number - [iteration] - is greater or equal to
  ///  some limit
  ///
  ///  - if passed [coefficientsDiff] is less than the minimum update value
  bool isConverged(double coefficientsDiff, int iteration);
}
