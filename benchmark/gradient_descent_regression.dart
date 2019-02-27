import 'dart:async';
import 'dart:typed_data';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/matrix.dart';

MLMatrix features;
MLMatrix labels;
LinearRegressor regressor;

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
    regressor = LinearRegressor.gradient(dtype: Float32x4);
  }

  void tearDown() {}
}

Future gradientDescentRegressionBenchmark() async {
  final data = MLData.fromCsvFile('datasets/advertising.csv',
      dtype: Float32x4, labelIdx: 3);
  features = await data.features;
  labels = await data.labels;

  GDRegressorBenchmark.main();
}
