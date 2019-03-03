import 'dart:async';
import 'dart:typed_data';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/matrix.dart';

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
    regressor.fit(features, labels);
  }

  @override
  void setup() {
    regressor = LinearClassifier.logisticRegressor(dtype: Float32x4);
  }

  void tearDown() {}
}

Future logisticRegressionBenchmark() async {
  final data = MLData.fromCsvFile('datasets/pima_indians_diabetes_database.csv',
      labelIdx: 8, dtype: Float32x4);
  features = await data.features;
  labels = await data.labels;

  LogisticRegressorBenchmark.main();
}
