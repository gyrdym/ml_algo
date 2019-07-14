import 'package:ml_algo/src/solver/linear/convergence_detector/convergence_detector.dart';

abstract class ConvergenceDetectorFactory {
  ConvergenceDetector create(double minUpdate, int iterationsLimit);
}
