import 'dart:io';
import 'dart:async';
import 'dart:convert';

import 'package:dart_ml/dart_ml.dart';
import 'package:csv/csv.dart' as csv;

main() async {
  csv.CsvCodec csvCodec = new csv.CsvCodec();
  Stream<List<int>> input = new File('example/advertising.csv').openRead();
  List<List<num>> fields = (await input.transform(UTF8.decoder)
      .transform(csvCodec.decoder).toList() as List<List<num>>)
      .sublist(1);

  List<double> extractFeatures(item) => item.sublist(0, 3)
      .map((num feature) => feature.toDouble())
      .toList();

  List<Vector> features = fields
      .map((List<num> item) => new Vector.from(extractFeatures(item)))
      .toList(growable: false);

  Vector labels = new Vector.from(fields.map((List<num> item) => item.last.toDouble()).toList());

  RMSEEstimator rmseEstimator = new RMSEEstimator();
  MAPEEstimator mapeEstimator = new MAPEEstimator();

  SGDLinearRegressor sgdRegressor = new SGDLinearRegressor();
  BGDLinearRegressor batchGdRegressor = new BGDLinearRegressor();
  MBGDLinearRegressor mbgdRegressor = new MBGDLinearRegressor();

  KFoldCrossValidator validator = new KFoldCrossValidator();

  print('K-fold cross validation:');
  print('\nRMSE:');
  print('SGD regressor: ${validator.validate(sgdRegressor, features, labels).mean()}');
  print('Batch GD regressor: ${validator.validate(batchGdRegressor, features, labels).mean()}');
  print('Mini batch GD regressor: ${validator.validate(mbgdRegressor, features, labels).mean()}');

  print('\nMAPE:');
  print('SGD GD regressor: ${validator.validate(sgdRegressor, features, labels, estimator: mapeEstimator).mean()}');
  print('Batch GD regressor: ${validator.validate(batchGdRegressor, features, labels, estimator: mapeEstimator).mean()}');
  print('Mini batch GD regressor: ${validator.validate(mbgdRegressor, features, labels, estimator: mapeEstimator).mean()}');
}
