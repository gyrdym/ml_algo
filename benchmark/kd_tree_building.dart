// 0.5 sec (MacBook Air mid 2017)
import 'dart:convert';
import 'dart:io';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

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
  final file = File('benchmark/data/sample_data.json');
  final dataAsString = await file.readAsString();
  final decoded = jsonDecode(dataAsString) as Map<String, dynamic>;

  trainData = DataFrame.fromJson(decoded);

  print(
      'Data dimension: ${trainData.rows.length}x${trainData.rows.first.length}');

  KDTreeBuildingBenchmark.main();
}
