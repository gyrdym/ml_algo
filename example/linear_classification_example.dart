import 'dart:io';
import 'dart:async';
import 'dart:convert';

import 'package:dart_ml/dart_ml.dart';
import 'package:csv/csv.dart' as csv;

Future main() async {
  Dependencies.configure();

  csv.CsvCodec csvCodec = new csv.CsvCodec();
  Stream<List<int>> input = new File('example/datasets/fertility.csv').openRead();
  List<List<num>> fields = (await input.transform(UTF8.decoder)
      .transform(csvCodec.decoder).toList() as List<List<num>>)
      .sublist(1);

  List<double> extractFeatures(item) =>
      item.map((Object feature) => (feature as num).toDouble()).toList();

  List<Float32x4Vector> features = fields
      .map((List item) => new Float32x4Vector.from(extractFeatures(item.sublist(0, item.length - 1))))
      .toList(growable: false);

  List<double> labels = fields
    .map((List item) => item.last == "N" ? 1.0 : 0.0)
    .toList(growable: false);

  print('features: $features');
  print('labels: $labels');

  LogisticRegressor logisticRegressor = new LogisticRegressor(learningRate: 1e-1);

  logisticRegressor.train(features.sublist(0,70), labels.sublist(0, 70));
  Float32x4Vector prediction = logisticRegressor.predictProbabilities(features.sublist(70));

  print('prediction: $prediction');
}