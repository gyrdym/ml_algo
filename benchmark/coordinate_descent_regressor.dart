import 'dart:convert';
import 'dart:io';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

const observationsNum = 1000;
const featuresNum = 100;
late DataFrame fittingData;

class CoordinateDescentRegressorBenchmark extends BenchmarkBase {
  CoordinateDescentRegressorBenchmark()
      : super('Linear regressor, coordinate descent');

  static void main() {
    CoordinateDescentRegressorBenchmark().report();
  }

  @override
  void run() {
    LinearRegressor(fittingData, 'col_100',
        optimizerType: LinearOptimizerType.coordinate, iterationsLimit: 30);
  }
}

Future main() async {
  final file = File('benchmark/data/sample_regression_data.json');
  final dataAsString = await file.readAsString();
  final decoded = jsonDecode(dataAsString) as Map<String, dynamic>;

  fittingData = DataFrame.fromJson(decoded);

  CoordinateDescentRegressorBenchmark.main();
}
