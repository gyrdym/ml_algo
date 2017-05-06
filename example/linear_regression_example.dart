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

  List<TypedVector> allFeatures = fields
      .map((List<num> item) => new TypedVector.from(extractFeatures(item)))
      .toList(growable: false);

  TypedVector allLabels = new TypedVector.from(fields.map((List<num> item) => item.last.toDouble()).toList());

  RMSEEstimator rmseEstimator = new RMSEEstimator();
  MAPEEstimator mapeEstimator = new MAPEEstimator();

  SGDLinearRegressor sgdRegressor = new SGDLinearRegressor();
  BGDLinearRegressor batchGdRegressor = new BGDLinearRegressor();
  MBGDLinearRegressor mbgdRegressor = new MBGDLinearRegressor();

  KFoldCrossValidator validator = new KFoldCrossValidator();

  print('SGD regressor: ${validator.validate(sgdRegressor, allFeatures, allLabels, rmseEstimator)}');
  print('Batch GD regressor: ${validator.validate(batchGdRegressor, allFeatures, allLabels, rmseEstimator)}');
  print('Mini batch GD regressor: ${validator.validate(mbgdRegressor, allFeatures, allLabels, rmseEstimator)}');

  print('SGD GD regressor: ${validator.validate(sgdRegressor, allFeatures, allLabels, mapeEstimator)}');
  print('Batch GD regressor: ${validator.validate(batchGdRegressor, allFeatures, allLabels, mapeEstimator)}');
  print('Mini batch GD regressor: ${validator.validate(mbgdRegressor, allFeatures, allLabels, mapeEstimator)}');
}
