// 0.04 sec (MacBook Air mid 2017)
import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/random_binary_projection_searcher_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/linalg.dart';

final k = 10;
final digitCapacity = 10;
final searchRadius = 3;

late DataFrame trainData;
late RandomBinaryProjectionSearcher searcher;
late Vector point;

class RandomBinaryProjectionSearcherQueryingBenchmark extends BenchmarkBase {
  RandomBinaryProjectionSearcherQueryingBenchmark()
      : super('RandomBinaryProjectionSearcher querying benchmark');

  static void main() {
    RandomBinaryProjectionSearcherQueryingBenchmark().report();
  }

  @override
  void run() {
    searcher.query(point, k, searchRadius);
  }

  void tearDown() {}
}

Future<dynamic> main() async {
  final points = Matrix.random(20000, 10, seed: 1, min: -5000, max: 5000);

  trainData = DataFrame.fromMatrix(points);
  searcher = RandomBinaryProjectionSearcher(trainData, digitCapacity, seed: 10);
  point = Vector.randomFilled(trainData.rows.first.length,
      seed: 10, min: -5000, max: 5000);

  print(
      'Data dimension: ${trainData.rows.length}x${trainData.rows.first.length}');
  print('Number of neighbours: $k');

  RandomBinaryProjectionSearcherQueryingBenchmark.main();

  print(
      'Amount of search iterations: ${(searcher as RandomBinaryProjectionSearcherImpl).searchIterationCount}');
}
