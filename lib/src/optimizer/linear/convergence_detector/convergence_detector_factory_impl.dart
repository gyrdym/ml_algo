import 'package:ml_algo/src/optimizer/linear/convergence_detector/convergence_detector.dart';
import 'package:ml_algo/src/optimizer/linear/convergence_detector/convergence_detector_factory.dart';
import 'package:ml_algo/src/optimizer/linear/convergence_detector/convergence_detector_impl.dart';

class ConvergenceDetectorFactoryImpl implements ConvergenceDetectorFactory {
  const ConvergenceDetectorFactoryImpl();

  @override
  ConvergenceDetector create(double minUpdate, int iterationsLimit) =>
      ConvergenceDetectorImpl(minUpdate, iterationsLimit);
}
