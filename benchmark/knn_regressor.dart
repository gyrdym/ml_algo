// 10.0 sec (MacBook Air mid 2017)
import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

const observationsNum = 500;
const featuresNum = 20;

class KnnRegressorBenchmark extends BenchmarkBase {
  KnnRegressorBenchmark() : super('Knn regression benchmark');

  Matrix features;
  DataFrame testFeatures;
  Matrix labels;
  Matrix testLabels;
  KnnRegressor regressor;


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

    testFeatures = DataFrame.fromMatrix(
        Matrix.fromRows(
            List.generate(
                observationsNum,
                (i) => Vector.randomFilled(featuresNum),
            ),
        ),
    );
    testLabels = Matrix.fromColumns([Vector.randomFilled(observationsNum)]);

    regressor = KnnRegressorImpl(features, labels, 'target', k: 7);
  }

  void tearDown() {}
}

Future main() async {
  KnnRegressorBenchmark.main();
}
