import 'dart:async';
import 'dart:typed_data';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/linalg.dart';

MLMatrix<Float32x4> features;
MLVector<Float32x4> labels;
GradientRegressor regressor;

class GDRegressorBenchmark extends BenchmarkBase {
  const GDRegressorBenchmark() : super('Gradient descent regressor');

  static void main() {
    const GDRegressorBenchmark().report();
  }

  @override
  void run() {
    regressor.fit(features, labels);
  }

  @override
  void setup() {
    regressor = GradientRegressor();
  }

  void tearDown() {}
}

Future gradientDescentRegression() async {
  final data = Float32x4CsvMLData.fromFile('datasets/advertising.csv', 3);
  features = await data.features;
  labels = await data.labels;

  GDRegressorBenchmark.main();
}
