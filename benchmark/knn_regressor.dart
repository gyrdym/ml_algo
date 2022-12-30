// Approx. 5.2 sec (Macbook Pro 2019, Dart 2.16.0)

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const observationsNum = 500;
const featuresNum = 20;

class KnnRegressorBenchmark extends BenchmarkBase {
  KnnRegressorBenchmark() : super('Knn regression benchmark');

  late DataFrame testFeatures;
  late KnnRegressor regressor;

  static void main() {
    KnnRegressorBenchmark().report();
  }

  @override
  void run() {
    regressor.predict(testFeatures);
  }

  @override
  void setup() {
    final featureMatrix = Matrix.fromRows(List.generate(
        observationsNum * 2, (i) => Vector.randomFilled(featuresNum, seed: 1)));

    final labelMatrix =
        Matrix.fromColumns([Vector.randomFilled(observationsNum * 2, seed: 2)]);

    final observations = DataFrame.fromMatrix(Matrix.fromColumns([
      ...featureMatrix.columns,
      ...labelMatrix.columns,
    ]));

    testFeatures = DataFrame.fromMatrix(
      Matrix.fromRows(
        List.generate(
          observationsNum,
          (i) => Vector.randomFilled(featuresNum, seed: 3),
        ),
      ),
    );

    regressor = KnnRegressor(observations, observations.header.last, 7);
  }

  void tearDown() {}
}

Future<void> main() async {
  KnnRegressorBenchmark.main();
}
