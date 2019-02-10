import 'dart:async';
import 'dart:typed_data';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/linear_classifier.dart';
import 'package:ml_algo/ml_data.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

MLMatrix features;
MLVector labels;
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
  final data = MLData.fromCsvFile('datasets/pima_indians_diabetes_database.csv', labelIdx: 8, dtype: Float32x4);
  features = await data.features;
  labels = await data.labels;

  LogisticRegressorBenchmark.main();
}
