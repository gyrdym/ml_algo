import 'dart:convert';
import 'dart:io';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

late DataFrame trainData;

class LassoRegressorBenchmark extends BenchmarkBase {
  LassoRegressorBenchmark() : super('Lasso regression, coordinate descent');

  static void main() {
    LassoRegressorBenchmark().report();
  }

  @override
  void run() {
    LinearRegressor.lasso(trainData, 'col_100', iterationLimit: 30);
  }
}

Future<void> main() async {
  final file = File('benchmark/data/sample_data.json');
  final dataAsString = await file.readAsString();
  final decoded = jsonDecode(dataAsString) as Map<String, dynamic>;

  trainData = DataFrame.fromJson(decoded);

  LassoRegressorBenchmark.main();
}
