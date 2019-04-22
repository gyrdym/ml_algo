// 10 sec
import 'dart:async';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const observationsNum = 1000;
const featuresNum = 20;

class CrossValidatorBenchmark extends BenchmarkBase {
  CrossValidatorBenchmark() : super('Cross validator benchmark');

  Matrix features;
  Matrix labels;
  CrossValidator crossValidator;

  static void main() {
    CrossValidatorBenchmark().report();
  }

  @override
  void run() {
    crossValidator.evaluate((trainFeatures, trainLabels) =>
        ParameterlessRegressor.knn(trainFeatures, trainLabels, k: 7),
        features, labels, MetricType.mape);
  }

  @override
  void setup() {
    features = Matrix.fromRows(List.generate(observationsNum,
            (i) => Vector.randomFilled(featuresNum)));
    labels = Matrix.fromColumns([Vector.randomFilled(observationsNum)]);

    crossValidator = CrossValidator.kFold(numberOfFolds: 5);
  }

  void tearDown() {}
}

Future main() async {
  CrossValidatorBenchmark.main();
}
