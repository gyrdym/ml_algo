// Approx. 5.5 second (MacBook Pro 2019), Dart version: 2.16.0

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const observationsNum = 4000;
const featuresNum = 100;

class LinearRegressorBenchmark extends BenchmarkBase {
  LinearRegressorBenchmark() : super('Linear regressor');

  late DataFrame fittingData;

  static void main() {
    LinearRegressorBenchmark().report();
  }

  @override
  void run() {
    LinearRegressor(fittingData, 'col_20');
  }

  @override
  void setup() {
    final features = Matrix.fromRows(List.generate(
        observationsNum, (i) => Vector.randomFilled(featuresNum)));

    final labels = Matrix.fromColumns([Vector.randomFilled(observationsNum)]);

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
  LinearRegressorBenchmark.main();
}
