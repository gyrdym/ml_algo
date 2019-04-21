// 5.7 sec
import 'dart:async';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const observationsNum = 500;
const featuresNum = 20;

class KnnRegressorBenchmark extends BenchmarkBase {
  KnnRegressorBenchmark() : super('KNN regression benchmark');

  Matrix features;
  Matrix testFeatures;
  Matrix labels;
  Matrix testLabels;
  ParameterlessRegressor regressor;


  static void main() {
    KnnRegressorBenchmark().report();
  }

  @override
  void run() {
    regressor.predict(testFeatures);
  }

  @override
  void setup() {
    features = Matrix.fromRows(List.generate(observationsNum * 2,
            (i) => Vector.randomFilled(featuresNum)));
    labels = Matrix.fromColumns([Vector.randomFilled(observationsNum * 2)]);

    testFeatures = Matrix.fromRows(List.generate(observationsNum,
            (i) => Vector.randomFilled(featuresNum)));
    testLabels = Matrix.fromColumns([Vector.randomFilled(observationsNum)]);

    regressor = ParameterlessRegressor.knn(features, labels, k: 7);
  }

  void tearDown() {}
}

Future main() async {
  KnnRegressorBenchmark.main();
}
