import 'dart:io';
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:ml_algo/ml_algo.dart';
import 'package:csv/csv.dart' as csv;

Future<double> logisticRegression() async {
  final csvCodec = csv.CsvCodec(eol: '\n');
  final input = File('example/datasets/pima_indians_diabetes_database.csv').openRead();
  final fields = (await input.transform(utf8.decoder).transform(csvCodec.decoder).toList()).sublist(1);

  List<double> extractFeatures(List<Object> item) => item.map((Object feature) => (feature as num).toDouble()).toList();

  final features = fields.map((List item) => extractFeatures(item.sublist(0, item.length - 1))).toList(growable: false);
  final featuresMatrix = Float32x4MatrixFactory.from(features);
  final labels = Float32x4VectorFactory.from(fields.map((List<dynamic> item) => (item.last as num).toDouble()));
  final validator = CrossValidator<Float32x4>.kFold(numberOfFolds: 7);
  final step = 0.001;
  final limit = 0.6;
  double minError = double.infinity;
//  double bestLearningRate = 0.0;
  for (double rate = step; rate < limit; rate += step) {
    final logisticRegressor = LogisticRegressor(
        iterationLimit: 100,
        learningRate: rate,
        batchSize: 1,
        learningRateType: LearningRateType.constant,
        fitIntercept: true);
    final error = validator.evaluate(logisticRegressor, featuresMatrix, labels, MetricType.accuracy);
    if (error < minError) {
      minError = error;
//      bestLearningRate = rate;
    }
  }

  return minError;
}
