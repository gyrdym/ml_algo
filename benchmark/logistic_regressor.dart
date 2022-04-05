// MacBook Air 13.3 mid 2017: ~ 4 sec

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';

const observationsNum = 20000;
const columnsNum = 101;

class LogisticRegressorBenchmark extends BenchmarkBase {
  LogisticRegressorBenchmark() : super('Logistic regressor');

  late DataFrame _data;

  static void main() {
    LogisticRegressorBenchmark().report();
  }

  @override
  void run() {
    LogisticRegressor(
      _data,
      'col_20',
      minCoefficientsUpdate: 1e-100000,
      iterationsLimit: 200,
    );
  }

  @override
  void setup() {
    final observations = Matrix.random(observationsNum, columnsNum, seed: 1);

    _data = DataFrame.fromMatrix(observations);
  }

  void tearDown() {}
}

Future main() async {
  LogisticRegressorBenchmark.main();
}
