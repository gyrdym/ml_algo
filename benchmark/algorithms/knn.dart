// MacBook Air 13.3 mid 2017: ~ 5 sec

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/src/knn_solver/knn_solver.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const k = 10;
const trainObservationsNum = 2000;
const observationsNum = 100;
const featuresNum = 100;

Matrix observations;
Matrix trainObservations;
Matrix labels;

class KnnBenchmark extends BenchmarkBase {
  const KnnBenchmark() : super('KNN benchmark');

  static void main() {
    const KnnBenchmark().report();
  }

  @override
  void run() {
    findKNeighbours(k, trainObservations, labels, observations)
        .toList(growable: false);
  }
}

void main() {
  trainObservations = Matrix.fromRows(List.generate(trainObservationsNum,
          (i) => Vector.randomFilled(featuresNum)));
  observations = Matrix.fromRows(List.generate(observationsNum,
          (i) => Vector.randomFilled(featuresNum)));
  labels = Matrix.fromColumns([Vector.randomFilled(trainObservationsNum)]);

  KnnBenchmark.main();
}
