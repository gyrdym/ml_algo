import 'dart:io';
import 'dart:async';
import 'dart:convert';

import 'package:dart_ml/dart_ml.dart';
import 'package:csv/csv.dart' as csv;

Future main() async {
  final csvCodec = csv.CsvCodec();
  final input = File('example/datasets/pima_indians_diabetes_database.csv').openRead();
  final fields = (await input.transform(utf8.decoder)
      .transform(csvCodec.decoder).toList() as List<List<num>>)
      .sublist(1);

  List<double> extractFeatures(List<Object> item) =>
      item.map((Object feature) => (feature as num).toDouble()).toList();

  final features = fields
      .map((List item) => Float32x4Vector.from(extractFeatures(item.sublist(0, item.length - 1))))
      .toList(growable: false);

  final labels = Float32x4Vector.from(fields.map((List<num> item) => item.last.toDouble()));
  final logisticRegressor = LogisticRegressor(iterationLimit: 100, learningRate: 0.0531, batchSize: 768,
    learningRateType: LearningRateType.constant, fitIntercept: true);
  final validator = CrossValidator<Float32x4Vector>.kFold(numberOfFolds: 7);

  print('Logistic regression, error on cross validation: ');
  print('${(validator.evaluate(logisticRegressor, features, labels, MetricType.ACCURACY) * 100).toStringAsFixed(2)}%');
}