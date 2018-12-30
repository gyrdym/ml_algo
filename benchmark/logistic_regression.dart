import 'dart:async';
import 'dart:typed_data';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/linalg.dart';

MLMatrix<Float32x4> features;
MLVector<Float32x4> labels;
LogisticRegressor regressor;

class LogisticRegressorBenchmark extends BenchmarkBase {
  const LogisticRegressorBenchmark() : super('Logistic regressor');

  static void main() {
    const LogisticRegressorBenchmark().report();
  }

  @override
  void run() {
    regressor.fit(features, labels);
  }

  @override
  void setup() {
    regressor = LogisticRegressor();
  }

  void tearDown() {}
}

Future logisticRegression() async {
  final data = Float32x4CsvMLData.fromFile('datasets/pima_indians_diabetes_database.csv', 8);
  features = await data.features;
  labels = await data.labels;

  LogisticRegressorBenchmark.main();
}
