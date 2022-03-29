// 0.03 sec (MacBook Air mid 2017)
import 'dart:convert';
import 'dart:io';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:ml_linalg/vector.dart';

final k = 10;

late DataFrame trainData;
late KDTree tree;
late Vector point;

class KDTreeQueryingBenchmark extends BenchmarkBase {
  KDTreeQueryingBenchmark() : super('KDTree querying benchmark');

  static void main() {
    KDTreeQueryingBenchmark().report();
  }

  @override
  void run() {
    tree.query(point, k);
  }

  void tearDown() {}
}

Future main() async {
  final file = File('benchmark/data/sample_data.json');
  final dataAsString = await file.readAsString();
  final decodedPoints = jsonDecode(dataAsString) as Map<String, dynamic>;

  trainData = DataFrame.fromJson(decodedPoints);
  tree = KDTree(trainData, leafSie: 1);
  point = Vector.randomFilled(trainData.rows.first.length,
      seed: 10, min: -5000, max: 5000);

  print(
      'Data dimension: ${trainData.rows.length}x${trainData.rows.first.length}');
  print('Number of neighbours: $k');

  KDTreeQueryingBenchmark.main();
}
