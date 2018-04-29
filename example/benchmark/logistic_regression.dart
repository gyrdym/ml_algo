// 3.134 sec

import 'dart:io';
import 'dart:async';
import 'dart:convert';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:dart_ml/dart_ml.dart';
import 'package:csv/csv.dart' as csv;

List<Float32x4Vector> features;
Float32x4Vector labels;
LogisticRegressor regressor;

class LogisticRegressorBenchmark extends BenchmarkBase {
  const LogisticRegressorBenchmark() : super('Logistic regressor');

  static void main() {
    new LogisticRegressorBenchmark().report();
  }

  void run() {
    regressor.fit(features, labels);
  }

  void setup() {
    regressor = new LogisticRegressor(batchSize: 1);
  }

  void tearDown() {}
}

Future main() async {
  final csvCodec = new csv.CsvCodec();
  Stream<List<int>> input = new File('example/datasets/pima_indians_diabetes_database.csv').openRead();
  List<List<num>> fields = (await input.transform(UTF8.decoder)
      .transform(csvCodec.decoder).toList() as List<List<num>>)
      .sublist(1);

  List<double> extractFeatures(item) =>
      item.map((Object feature) => (feature as num).toDouble()).toList();

  features = fields
      .map((List item) => new Float32x4Vector.from(extractFeatures(item.sublist(0, item.length - 1))))
      .toList(growable: false);
  labels = new Float32x4Vector.from(fields.map((List<num> item) => item.last.toDouble()));

  LogisticRegressorBenchmark.main();
}
