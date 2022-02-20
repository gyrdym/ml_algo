import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';

const observationsNum = 1000;
const featuresNum = 100;

class CoordinateDescentRegressorBenchmark extends BenchmarkBase {
  CoordinateDescentRegressorBenchmark()
      : super('Linear regressor, coordinate descent');

  late DataFrame fittingData;

  static void main() {
    CoordinateDescentRegressorBenchmark().report();
  }

  @override
  void run() {
    LinearRegressor(fittingData, 'col_100',
        optimizerType: LinearOptimizerType.coordinate, iterationsLimit: 30);
  }

  @override
  void setup() {
    final features = Matrix.random(observationsNum, featuresNum);
    final labels = Matrix.random(observationsNum, 1);

    fittingData = DataFrame.fromMatrix(
      Matrix.fromColumns([
        ...features.columns,
        ...labels.columns,
      ]),
    );
  }

  void tearDown() {}
}

Future main() async {
  CoordinateDescentRegressorBenchmark.main();
}
