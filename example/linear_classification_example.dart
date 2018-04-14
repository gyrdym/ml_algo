import 'dart:io';
import 'dart:async';
import 'dart:convert';

import 'package:dart_ml/dart_ml.dart';
import 'package:csv/csv.dart' as csv;

Future main() async {
  csv.CsvCodec csvCodec = new csv.CsvCodec();
  Stream<List<int>> input = new File('example/datasets/pima_indians_diabetes_database.csv').openRead();
  List<List<num>> fields = (await input.transform(UTF8.decoder)
      .transform(csvCodec.decoder).toList() as List<List<num>>)
      .sublist(1);

  List<double> extractFeatures(item) =>
      item.map((Object feature) => (feature as num).toDouble()).toList();

  List<Float32x4Vector> features = fields
      .map((List item) => new Float32x4Vector.from(extractFeatures(item.sublist(0, item.length - 1))))
      .toList(growable: false);

  List<double> labels = fields.map((List<num> item) => item.last * 1.0).toList(growable: false);

  final logisticRegressor = new LogisticRegressor(alpha: 0.0);
  final validator = new CrossValidator.KFold();

  print('Ratio of incorrect answers on a cross validation: ');
  print(validator.evaluate(logisticRegressor, features, labels).mean());
}