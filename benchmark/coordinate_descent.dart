// Approx. 0.06 sec (Macbook Pro 2019, Dart 3.12.2)

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/src/linear_optimizer/convergence_detector/convergence_detector_impl.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/zero_coefficients_generator.dart';
import 'package:ml_algo/src/linear_optimizer/least_squares_coordinate_descent_optimizer.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const observationsNum = 10000;
const featuresNum = 20;
const iterationsLimit = 10;

class CoordinateDescentBenchmark extends BenchmarkBase {
  CoordinateDescentBenchmark() : super('Coordinate descent optimizer');

  late Matrix features;
  late Matrix labels;

  static void main() {
    CoordinateDescentBenchmark().report();
  }

  @override
  void run() {
    final optimizer = LeastSquaresCoordinateDescentOptimizer(
      features,
      labels,
      dtype: DType.float32,
      lambda: 0.01,
      initialCoefficientsGenerator: ZeroCoefficientsGenerator(DType.float32),
      convergenceDetector: ConvergenceDetectorImpl(1e-5, iterationsLimit),
      isFittingDataNormalized: false,
    );

    optimizer.findExtrema();
  }

  @override
  void setup() {
    features = Matrix.fromRows(
      List.generate(
        observationsNum,
        (i) => Vector.randomFilled(featuresNum, seed: i),
      ),
      dtype: DType.float32,
    );
    labels = Matrix.fromColumns(
      [Vector.randomFilled(observationsNum, seed: observationsNum)],
      dtype: DType.float32,
    );
  }
}

void main() {
  CoordinateDescentBenchmark.main();
}
