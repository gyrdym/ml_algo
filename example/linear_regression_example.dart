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

  SGDLinearRegressor predictor = new SGDLinearRegressor<RegularVector>();

  List<RegularVector> features = fields
      .map((List<num> item) => new RegularVector.from(extractFeatures(item)))
      .toList(growable: false);

  RegularVector labels = new RegularVector.from(fields.map((List<num> item) => item.last.toDouble()).toList());

  predictor.train(features, labels);

  print("weights: ${predictor.weights}");
  print("rmse (training) is: ${predictor.rmse}");
}
