import 'dart:async';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const observationsNum = 200;
const featuresNum = 20;

class LogisticRegressorBenchmark extends BenchmarkBase {
  LogisticRegressorBenchmark() : super('Logistic regressor');

  final Matrix features = Matrix.fromRows(List.generate(observationsNum,
    (i) => Vector.randomFilled(featuresNum)));
  final Matrix labels = Matrix.fromColumns([Vector.zero(observationsNum)]);

  static void main() {
    LogisticRegressorBenchmark().report();
  }

  @override
  void run() {
    LogisticRegressor.gradient(features, labels,
        dtype: DType.float32, minWeightsUpdate: null, iterationsLimit: 200);
  }

  void tearDown() {}
}

Future main() async {
  LogisticRegressorBenchmark.main();
}
