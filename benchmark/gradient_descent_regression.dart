import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:csv/csv.dart' as csv;

List<List<double>> features;
MLVector<Float32x4> labels;
GradientRegressor regressor;

class GDRegressorBenchmark extends BenchmarkBase {
  const GDRegressorBenchmark() : super('Gradient descent regressor');

  static void main() {
    const GDRegressorBenchmark().report();
  }

  @override
  void run() {
    regressor.fit(Float32x4MatrixFactory.from(features), labels);
  }

  @override
  void setup() {
    regressor = GradientRegressor();
  }

  void tearDown() {}
}

Future gradientDescentRegression() async {
  final csvCodec = csv.CsvCodec(eol: '\n');
  final input = File('example/datasets/advertising.csv').openRead();
  final fields = (await input.transform(utf8.decoder).transform(csvCodec.decoder).toList()).sublist(1);

  List<double> extractFeatures(List item) =>
      item.sublist(0, 3).map<double>((dynamic feature) => (feature as num).toDouble()).toList();

  features = fields.map<List<double>>(extractFeatures).toList(growable: false);

  labels = Float32x4VectorFactory.from(fields.map((List<dynamic> item) => (item.last as num).toDouble()));

  GDRegressorBenchmark.main();
}
