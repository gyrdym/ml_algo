import 'package:ml_linalg/vector.dart';

/**
 * An entity, that decides, when to stop an optimization algorithm
 *
 * The criteria of convergence - if the algorithm hits one of the limits:
 *  - if the passed iteration number - [iteration] - is greater or equal to some limit
 *  - if one of the passed coefficient updates - [coefficientUpdates] - is less than a minimum update value
 */
abstract class ConvergenceDetector {
  double get minUpdate;
  int get iterationsLimit;

  bool isConverged(MLVector coefficientUpdates, int iteration);
}
