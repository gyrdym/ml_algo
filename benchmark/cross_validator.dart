// 8.5 sec
import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const observationsNum = 1000;
const columnsNum = 21;

class CrossValidatorBenchmark extends BenchmarkBase {
  CrossValidatorBenchmark() : super('Cross validator benchmark');

  CrossValidator crossValidator;

  static void main() {
    CrossValidatorBenchmark().report();
  }

  @override
  void run() {
    crossValidator.evaluate((trainSamples, targetFeatureNames) =>
        KnnRegressor(trainSamples, targetFeatureNames.first, k: 7),
        MetricType.mape);
  }

  @override
  void setup() {
    final samples = Matrix.fromRows(List.generate(observationsNum,
            (i) => Vector.randomFilled(columnsNum)));

    final dataFrame = DataFrame.fromMatrix(samples);

    crossValidator = CrossValidator.kFold(dataFrame, ['col_20'],
        numberOfFolds: 5);
  }

  void tearDown() {}
}

Future main() async {
  CrossValidatorBenchmark.main();
}
