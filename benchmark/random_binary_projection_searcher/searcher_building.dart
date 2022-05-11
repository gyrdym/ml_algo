// 0.03 sec (MacBook Air mid 2017)
import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';

late DataFrame trainData;

class RandomBinaryProjectionSearcherBuildingBenchmark extends BenchmarkBase {
  RandomBinaryProjectionSearcherBuildingBenchmark()
      : super('RandomBinaryProjectionSearcher building benchmark');

  static void main() {
    RandomBinaryProjectionSearcherBuildingBenchmark().report();
  }

  @override
  void run() {
    RandomBinaryProjectionSearcher(trainData, 4, seed: 10);
  }

  void tearDown() {}
}

Future main() async {
  final points = Matrix.random(1000, 10, seed: 1, min: -5000, max: 5000);

  trainData = DataFrame.fromMatrix(points);

  print(
      'Data dimension: ${trainData.rows.length}x${trainData.rows.first.length}');

  RandomBinaryProjectionSearcherBuildingBenchmark.main();
}
