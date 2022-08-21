import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const observationsNum = 20000;
const featuresNum = 100;

class SGDRegressorBenchmark extends BenchmarkBase {
  SGDRegressorBenchmark() : super('SGD regressor');

  late DataFrame trainData;

  static void main() {
    SGDRegressorBenchmark().report();
  }

  @override
  void run() {
    LinearRegressor.SGD(trainData, 'col_100');
  }

  @override
  void setup() {
    final features = Matrix.fromRows(List.generate(
        observationsNum, (i) => Vector.randomFilled(featuresNum)));

    final labels = Matrix.fromColumns([Vector.randomFilled(observationsNum)]);

    trainData = DataFrame.fromMatrix(
      Matrix.fromColumns([
        ...features.columns,
        ...labels.columns,
      ]),
    );
  }

  void tearDown() {}
}

Future<dynamic> main() async {
  SGDRegressorBenchmark.main();
}
