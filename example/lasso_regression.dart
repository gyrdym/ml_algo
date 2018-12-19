import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:csv/csv.dart' as csv;
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/linalg.dart';

Future<double> lassoRegression() async {
  final csvCodec = csv.CsvCodec(eol: '\n');
  final input = File('datasets/advertising.csv').openRead();
  final fields = (await input.transform(utf8.decoder).transform(csvCodec.decoder).toList()).sublist(1);

  List<double> extractFeatures(List<dynamic> item) =>
      item.sublist(0, 3).map((dynamic feature) => (feature as num).toDouble()).toList();

  final features = fields.map(extractFeatures).toList(growable: false);

  final labels = Float32x4Vector.from(fields.map((List<dynamic> item) => (item.last as num).toDouble()));
  final lassoRegressionModel = LassoRegressor(iterationLimit: 100, lambda: 6750.0);
  final validator = CrossValidator.kFold();

  return validator.evaluate(lassoRegressionModel, Float32x4Matrix.from(features), labels, MetricType.mape);
}
