// Approx. 5.1 sec (Macbook Pro 2019, Dart 2.16.0)

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_impl.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const k = 10;
const trainObservationsNum = 2000;
const observationsNum = 100;
const featuresNum = 100;

class KnnSolverBenchmark extends BenchmarkBase {
  KnnSolverBenchmark() : super('KnnSolver benchmark');

  late KnnSolver solver;
  late Matrix features;

  static void main() {
    KnnSolverBenchmark().report();
  }

  @override
  void run() {
    solver.findKNeighbours(features).toList(growable: false);
  }

  @override
  void setup() {
    final trainFeatures = Matrix.fromRows(List.generate(
        trainObservationsNum, (i) => Vector.randomFilled(featuresNum, seed: 1)));
    final trainLabels =
        Matrix.fromColumns([Vector.randomFilled(trainObservationsNum, seed: 2)]);

    solver =
        KnnSolverImpl(trainFeatures, trainLabels, k, Distance.euclidean, false);

    features = Matrix.fromRows(List.generate(
        observationsNum, (i) => Vector.randomFilled(featuresNum, seed: 3)));
  }
}

void main() {
  KnnSolverBenchmark.main();
}
