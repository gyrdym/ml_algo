// 0.02346 sec (MacBook Air mid 2017)

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:dart_ml/dart_ml.dart';
import 'package:csv/csv.dart' as csv;

List<Vector<Float32x4>> features;
Vector<Float32x4> labels;
GradientRegressor regressor;

class GDRegressorBenchmark extends BenchmarkBase {
  const GDRegressorBenchmark() : super('Gradient descent regressor');

  static void main() {
    const GDRegressorBenchmark().report();
  }

  @override
  void run() {
    regressor.fit(features, labels);
  }

  @override
  void setup() {
    regressor = GradientRegressor();
  }

  void tearDown() {}
}

Future main() async {
  final csvCodec = csv.CsvCodec(eol: '\n');
  final input = File('example/datasets/advertising.csv').openRead();
  final fields = (await input.transform(utf8.decoder)
      .transform(csvCodec.decoder).toList())
      .sublist(1);

  List<double> extractFeatures(List item) => item.sublist(0, 3)
      .map<double>((dynamic feature) => (feature as num).toDouble())
      .toList();

  features = fields
      .map<Vector<Float32x4>>((List item) => Float32x4VectorFactory.from(extractFeatures(item)))
      .toList(growable: false);

  labels = Float32x4VectorFactory.from(fields.map((List<dynamic> item) => (item.last as num).toDouble()));

  GDRegressorBenchmark.main();
}