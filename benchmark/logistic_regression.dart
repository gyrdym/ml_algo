import 'dart:async';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const observationsNum = 200;
const columnsNum = 21;

class LogisticRegressorBenchmark extends BenchmarkBase {
  LogisticRegressorBenchmark() : super('Logistic regressor');

  DataFrame _data;

  static void main() {
    LogisticRegressorBenchmark().report();
  }

  @override
  void run() {
    LogisticRegressor(_data, 'col_20',
        dtype: DType.float32, minWeightsUpdate: null, iterationsLimit: 200);
  }

  @override
  void setup() {
    final Matrix observations = Matrix.fromRows(List.generate(observationsNum,
            (i) => Vector.randomFilled(columnsNum)));

    _data = DataFrame.fromMatrix(observations);
  }

  void tearDown() {}
}

Future main() async {
  LogisticRegressorBenchmark.main();
}
