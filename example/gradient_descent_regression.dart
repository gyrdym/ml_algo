import 'dart:async';
import 'dart:io';
import 'dart:convert';

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
      .map((List<num> item) => Float32x4Vector.from(extractFeatures(item)))
      .toList(growable: false);

  final labels = Float32x4Vector.from(fields.map((List<num> item) => item.last.toDouble()));
  final sgdRegressionModel = GradientRegressor(type: GradientType.Stochastic, iterationLimit: 100000,
    learningRate: 1e-5, learningRateType: LearningRateType.constant);
  final validator = CrossValidator<Float32x4Vector>.KFold();

  print('K-fold cross validation with MAPE metric (error in percents):');
  print('${(validator.evaluate(sgdRegressionModel, features, labels, MetricType.MAPE)).toStringAsFixed(2)}%');
}
