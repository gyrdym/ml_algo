import 'dart:async';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const observationsNum = 200;
const featuresNum = 20;

class GDRegressorBenchmark extends BenchmarkBase {
  GDRegressorBenchmark() : super('Gradient descent regressor');

  final Matrix features = Matrix.fromRows(List.generate(observationsNum,
          (i) => Vector.randomFilled(featuresNum)));

  final Matrix labels = Matrix.fromColumns([Vector.randomFilled(observationsNum)]);

  static void main() {
    GDRegressorBenchmark().report();
  }

  @override
  void run() {
    LinearRegressor.gradient(features, labels, dtype: DType.float32);
  }

  void tearDown() {}
}

Future main() async {
  GDRegressorBenchmark.main();
}
