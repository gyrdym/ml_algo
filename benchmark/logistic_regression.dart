import 'dart:async';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const observationsNum = 200;
const featuresNum = 20;

Matrix features;
Matrix labels;
LinearClassifier regressor;

class LogisticRegressorBenchmark extends BenchmarkBase {
  const LogisticRegressorBenchmark() : super('Logistic regressor');

  static void main() {
    const LogisticRegressorBenchmark().report();
  }

  @override
  void run() {
    regressor.fit();
  }

  @override
  void setup() {
    regressor = LinearClassifier.logisticRegressor(features, labels,
        dtype: DType.float32, minWeightsUpdate: null, iterationsLimit: 200);
  }

  void tearDown() {}
}

Future main() async {
  features = Matrix.fromRows(List.generate(observationsNum,
          (i) => Vector.randomFilled(featuresNum)));
  labels = Matrix.fromColumns([Vector.zero(observationsNum)]);
  LogisticRegressorBenchmark.main();
}
