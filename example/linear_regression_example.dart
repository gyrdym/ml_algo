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

  SGDLinearRegressor predictor = new SGDLinearRegressor();

  List<List<double>> features = fields
      .map((List<num> item) => item.sublist(1, 3)
      .map((num feature) => feature.toDouble()).toList())
      .toList();

  List<double> labels = fields
      .map((List<num> item) => item.last.toDouble())
      .toList();

  predictor.train(features, labels);

  print("weights: ${predictor.weights}");
  print("rmse (training) is: ${predictor.rmse}");
}
