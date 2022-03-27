// 0.5 sec (MacBook Air mid 2017)
import 'dart:convert';
import 'dart:io';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

const observationsNum = 500;
const featuresNum = 20;
late DataFrame trainData;

class KDTreeBenchmark extends BenchmarkBase {
  KDTreeBenchmark() : super('KDTree benchmark');

  static void main() {
    KDTreeBenchmark().report();
  }

  @override
  void run() {
    KDTree(trainData);
  }

  void tearDown() {}
}

Future main() async {
  final file = File('benchmark/data/sample_regression_data.json');
  final dataAsString = await file.readAsString();
  final decoded = jsonDecode(dataAsString) as Map<String, dynamic>;

  trainData = DataFrame.fromJson(decoded);

  print('Data dimension: ${trainData.rows.length}x${trainData.rows.first.length}');

  KDTreeBenchmark.main();
}
