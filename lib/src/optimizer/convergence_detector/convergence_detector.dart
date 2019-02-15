import 'package:ml_linalg/vector.dart';

abstract class ConvergenceDetector {
  double get minUpdate;
  int get iterationsLimit;

  bool isConverged(MLVector coefficientUpdates, int iteration);
}
