import 'dart:io';
import 'dart:async';
import 'dart:convert';

import 'package:dart_ml/dart_ml.dart';
import 'package:csv/csv.dart' as csv;

main() async {
  csv.CsvCodec csvCodec = new csv.CsvCodec();
  Stream<List<int>> input = new File('example/datasets/advertising.csv').openRead();
  List<List<num>> fields = (await input.transform(UTF8.decoder)
      .transform(csvCodec.decoder).toList() as List<List<num>>)
      .sublist(1);

  List<double> extractFeatures(item) => item.sublist(0, 3)
      .map((num feature) => feature.toDouble())
      .toList();

  List<Float32x4Vector> features = fields
      .map((List<num> item) => new Float32x4Vector.from(extractFeatures(item)))
      .toList(growable: false);

  List<double> labels = fields.map((List<num> item) => item.last.toDouble()).toList();

  SGDRegressor sgdRegressor = new SGDRegressor();
  CrossValidatorImpl validator = new CrossValidatorImpl.KFold();

  print('K-fold cross validation:');
  print('\nRMSE:');
  print('SGD regressor: ${validator.evaluate(sgdRegressor, features, labels).mean()}');

  print('\nMAPE:');
  print('SGD GD regressor: ${validator.evaluate(sgdRegressor, features, labels).mean()}');
}
