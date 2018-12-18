import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:csv/csv.dart' as csv;
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/linalg.dart';

Future<double> gradientDescentRegression() async {
  final csvCodec = csv.CsvCodec(eol: '\n');
  final input = File('datasets/advertising.csv').openRead();
  final fields = (await input.transform(utf8.decoder).transform(csvCodec.decoder).toList()).sublist(1);

  List<double> extractFeatures(List<dynamic> item) =>
      item.sublist(0, 3).map((dynamic feature) => (feature as num).toDouble()).toList();

  final features = fields.map(extractFeatures).toList(growable: false);

  final labels = Float32x4Vector.from(fields.map((List<dynamic> item) => (item.last as num).toDouble()));
  final sgdRegressionModel = GradientRegressor(
      type: GradientType.stochastic,
      iterationLimit: 100000,
      learningRate: 1e-5,
      learningRateType: LearningRateType.constant);
  final validator = CrossValidator<Float32x4>.kFold();
  final quality =
      validator.evaluate(sgdRegressionModel, Float32x4Matrix.from(features), labels, MetricType.mape);

  return quality;
}
