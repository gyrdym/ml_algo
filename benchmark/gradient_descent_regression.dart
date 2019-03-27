import 'dart:async';
import 'dart:typed_data';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const observationsNum = 200;
const featuresNum = 20;

Matrix features;
Matrix labels;
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
  features = Matrix.rows(List.generate(observationsNum,
          (i) => Vector.randomFilled(featuresNum)));
  labels = Matrix.columns([Vector.randomFilled(observationsNum)]);

  GDRegressorBenchmark.main();
}
