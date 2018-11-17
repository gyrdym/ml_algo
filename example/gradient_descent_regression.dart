import 'dart:async';
import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';

import 'package:dart_ml/dart_ml.dart';
import 'package:csv/csv.dart' as csv;

Future main() async {
  final csvCodec = csv.CsvCodec(eol: '\n');
  final input = File('example/datasets/advertising.csv').openRead();
  final fields = (await input.transform(utf8.decoder)
      .transform(csvCodec.decoder).toList())
      .sublist(1);

  List<double> extractFeatures(List<dynamic> item) => item.sublist(0, 3)
      .map((dynamic feature) => (feature as num).toDouble())
      .toList();

  final features = fields
      .map((List<dynamic> item) => Float32x4VectorFactory.from(extractFeatures(item)))
      .toList(growable: false);

  final labels = Float32x4VectorFactory.from(fields.map((List<dynamic> item) => (item.last as num).toDouble()));
  final sgdRegressionModel = GradientRegressor(type: GradientType.stochastic, iterationLimit: 100000,
    learningRate: 1e-5, learningRateType: LearningRateType.constant);
  final validator = CrossValidator<Float32x4>.kFold();

  print('K-fold cross validation with MAPE metric (error in percents):');
  print('${(validator.evaluate(sgdRegressionModel, features, labels, MetricType.mape)).toStringAsFixed(2)}%');
}
