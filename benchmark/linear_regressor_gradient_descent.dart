// Approx. 0.4 second (MacBook Pro 2019, Dart version: 2.16.0)

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const observationsNum = 4000;
const featuresNum = 100;

class LinearRegressorGradientDescentBenchmark extends BenchmarkBase {
  LinearRegressorGradientDescentBenchmark()
      : super('Linear regressor gradient descent');

  late DataFrame fittingData;

  static void main() {
    LinearRegressorGradientDescentBenchmark().report();
  }

  @override
  void run() {
    LinearRegressor.SGD(fittingData, 'col_20', randomSeed: 12);
  }

  @override
  void setup() {
    final features = Matrix.fromRows(List.generate(
        observationsNum, (i) => Vector.randomFilled(featuresNum, seed: 12)));

    final labels =
        Matrix.fromColumns([Vector.randomFilled(observationsNum, seed: 13)]);

    fittingData = DataFrame.fromMatrix(
      Matrix.fromColumns([
        ...features.columns,
        ...labels.columns,
      ]),
    );
  }

  void tearDown() {}
}

Future<void> main() async {
  LinearRegressorGradientDescentBenchmark.main();
}
