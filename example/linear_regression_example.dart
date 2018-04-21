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

//  final sumOfAllFeatures = features.reduce((final combine, final vector) => combine + vector);
//  final normalaziedFeatures = features.map((final vector) => vector / sumOfAllFeatures).toList(growable: false);

  final labels = fields.map((List<num> item) => item.last.toDouble()).toList();
//  final sumOfAllLabels = labels.reduce((final combine, final label) => combine + label);
//  final normalizedLabels = labels.map((final label) => label / sumOfAllLabels).toList(growable: false);

  final sgdRegressionModel = new GradientRegressor();
  final lassoRegressionModel = new LassoRegressor(lambda: 1000.0);
  final validator = new CrossValidator<Float32x4Vector>.KFold();

  print('K-fold cross validation:');
  print('\nRMSE:');
  print('SGD regressor: ${validator.evaluate(sgdRegressionModel, features, labels, MetricType.RMSE)}');
  print('Lasso regressor: ${validator.evaluate(lassoRegressionModel, features, labels, MetricType.RMSE)}');

  print('\nMAPE:');
  print('SGD regressor: ${validator.evaluate(sgdRegressionModel, features, labels, MetricType.MAPE)}');
  print('Lasso regressor: ${validator.evaluate(lassoRegressionModel, features, labels, MetricType.MAPE)}');
}
