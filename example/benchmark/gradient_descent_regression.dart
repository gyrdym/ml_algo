// 0.161 sec

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:dart_ml/dart_ml.dart';
import 'package:csv/csv.dart' as csv;

List<Float32x4Vector> features;
Float32x4Vector labels;
GradientRegressor regressor;

class GDRegressorBenchmark extends BenchmarkBase {
  const GDRegressorBenchmark() : super('Gradient descent regressor');

  static void main() {
    new GDRegressorBenchmark().report();
  }

  void run() {
    regressor.fit(features, labels);
  }

  void setup() {
    regressor = new GradientRegressor();
  }

  void tearDown() {}
}

Future main() async {
  final csvCodec = new csv.CsvCodec();
  final input = new File('example/datasets/advertising.csv').openRead();
  final fields = (await input.transform(UTF8.decoder)
      .transform(csvCodec.decoder).toList() as List<List<num>>)
      .sublist(1);

  List<double> extractFeatures(item) => item.sublist(0, 3)
      .map((num feature) => feature.toDouble())
      .toList();

  features = fields
      .map((List<num> item) => new Float32x4Vector.from(extractFeatures(item)))
      .toList(growable: false);

  labels = new Float32x4Vector.from(fields.map((List<num> item) => item.last.toDouble()));

  GDRegressorBenchmark.main();
}