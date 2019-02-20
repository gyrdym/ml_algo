/**
 * An entity, that decides, when to stop an optimization algorithm
 *
 * The criteria of convergence - if the algorithm hits one of the limits:
 *  - if the passed iteration number - [iteration] - is greater or equal to some limit
 *  - if passed [coefficientsDiff] is less than a minimum update value
 */
abstract class ConvergenceDetector {
  double get minDiff;
  int get iterationsLimit;

  bool isConverged(double coefficientsDiff, int iteration);
}
