// 3.134 sec

import 'dart:io';
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:dart_ml/dart_ml.dart';
import 'package:csv/csv.dart' as csv;

List<SIMDVector<Float32x4List, Float32List, Float32x4>> features;
SIMDVector<Float32x4List, Float32List, Float32x4> labels;
LogisticRegressor regressor;

class LogisticRegressorBenchmark extends BenchmarkBase {
  const LogisticRegressorBenchmark() : super('Logistic regressor');

  static void main() {
    const LogisticRegressorBenchmark().report();
  }

  @override
  void run() {
    regressor.fit(features, labels);
  }

  @override
  void setup() {
    regressor = LogisticRegressor();
  }

  void tearDown() {}
}

Future main() async {
  final csvCodec = csv.CsvCodec();
  final input = File('example/datasets/pima_indians_diabetes_database.csv').openRead();
  final fields = (await input.transform(utf8.decoder)
      .transform(csvCodec.decoder).toList() as List<List<num>>)
      .sublist(1);

  List<double> extractFeatures(List<Object> item) =>
      item.map((Object feature) => (feature as num).toDouble()).toList();

  features = fields
      .map((List item) => Float32x4VectorFactory.from(extractFeatures(item.sublist(0, item.length - 1))))
      .toList(growable: false);
  labels = Float32x4VectorFactory.from(fields.map((List<num> item) => item.last.toDouble()));

  LogisticRegressorBenchmark.main();
}
