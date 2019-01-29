import 'dart:async';
import 'dart:typed_data';

import 'package:ml_algo/cross_validator.dart';
import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/linear_classifier.dart';
import 'package:ml_algo/metric_type.dart';
import 'package:ml_algo/ml_data.dart';

Future main() async {
  final data = MLData.fromCsvFile('datasets/pima_indians_diabetes_database.csv', labelIdx: 8, dtype: Float32x4);

  final features = await data.features;
  final labels = await data.labels;

  final validator = CrossValidator.kFold(numberOfFolds: 5, dtype: Float32x4);

  final step = 0.00001;
  final start = 0.001;
  final limit = 0.01;

  double minError = double.infinity;
  double bestLearningRate = 0.0;

  // Let's find optimal learningRate. WARNING: it may take very much time!
  for (double rate = start; rate < limit; rate += step) {
    final logisticRegressor = LinearClassifier.logisticRegressor(
        iterationLimit: 100,
        learningRate: rate,
        learningRateType: LearningRateType.constant,
        batchSize: 768,
        fitIntercept: true);
    final error = validator.evaluate(logisticRegressor, features, labels, MetricType.accuracy);
    if (error < minError) {
      minError = error;
      bestLearningRate = rate;
      print('error: $minError, learning rate: $bestLearningRate');
    }
  }
}
