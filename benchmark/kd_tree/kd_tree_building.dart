// 0.14 sec (MacBook Air mid 2017)
import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';

late DataFrame trainData;

class KDTreeBuildingBenchmark extends BenchmarkBase {
  KDTreeBuildingBenchmark() : super('KDTree building benchmark');

  static void main() {
    KDTreeBuildingBenchmark().report();
  }

  @override
  void run() {
    KDTree(trainData);
  }

  void tearDown() {}
}

Future main() async {
  final points = Matrix.random(1000, 10, seed: 1, min: -5000, max: 5000);

  trainData = DataFrame.fromMatrix(points);

  print(
      'Data dimension: ${trainData.rows.length}x${trainData.rows.first.length}');

  KDTreeBuildingBenchmark.main();
}
