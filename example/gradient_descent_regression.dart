import 'dart:io';
import 'dart:convert';

import 'package:dart_ml/dart_ml.dart';
import 'package:csv/csv.dart' as csv;

main() async {
  final csvCodec = new csv.CsvCodec();
  final input = new File('example/datasets/advertising.csv').openRead();
  final fields = (await input.transform(UTF8.decoder)
      .transform(csvCodec.decoder).toList() as List<List<num>>)
      .sublist(1);

  List<double> extractFeatures(item) => item.sublist(0, 3)
      .map((num feature) => feature.toDouble())
      .toList();

  final features = fields
      .map((List<num> item) => new Float32x4Vector.from(extractFeatures(item)))
      .toList(growable: false);

  final labels = new Float32x4Vector.from(fields.map((List<num> item) => item.last.toDouble()));
  final sgdRegressionModel = new GradientRegressor();
  final validator = new CrossValidator<Float32x4Vector>.KFold();

  print('K-fold cross validation with MAPE metric (percent error):');
  print('${(validator.evaluate(sgdRegressionModel, features, labels, MetricType.MAPE)).toStringAsFixed(2)}%');
}
