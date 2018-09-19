import 'dart:async';
import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';

import 'package:dart_ml/dart_ml.dart';
import 'package:csv/csv.dart' as csv;

Future main() async {
  final csvCodec = csv.CsvCodec();
  final input = File('example/datasets/advertising.csv').openRead();
  final fields = (await input.transform(utf8.decoder)
      .transform(csvCodec.decoder).toList() as List<List<num>>)
      .sublist(1);

  List<double> extractFeatures(List<num> item) => item.sublist(0, 3)
      .map((num feature) => feature.toDouble())
      .toList();

  final features = fields
      .map((List<num> item) => Float32x4VectorFactory.from(extractFeatures(item)))
      .toList(growable: false);

  final labels = Float32x4VectorFactory.from(fields.map((List<num> item) => item.last.toDouble()));
  final lassoRegressionModel = LassoRegressor(iterationLimit: 100, lambda: 6750.0);
  final validator = CrossValidator<Float32x4List, Float32List, Float32x4>.kFold();

  print('K-fold cross validation with MAPE metric:');
  print('Lasso regressor: ${validator.evaluate(lassoRegressionModel, features, labels, MetricType.MAPE)}');

  print('Feature weights (possibly, some weights are downgraded to zero, cause it is an objective of Lasso Regression):');
  print(lassoRegressionModel.weights);
}
