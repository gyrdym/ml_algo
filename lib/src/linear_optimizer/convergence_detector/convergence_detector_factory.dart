import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector.dart';

abstract class ConvergenceDetectorFactory {
  ConvergenceDetector create(double minUpdate, int iterationsLimit);
}
