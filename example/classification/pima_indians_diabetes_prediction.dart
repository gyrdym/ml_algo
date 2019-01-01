import 'dart:async';

import 'package:ml_algo/float32x4_cross_validator.dart';
import 'package:ml_algo/float32x4_csv_ml_data.dart';
import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/logistic_regressor.dart';
import 'package:ml_algo/metric_type.dart';

Future<double> logisticRegression() async {
  final data = Float32x4CsvMLData.fromFile('datasets/pima_indians_diabetes_database.csv', labelIdx: 8);

  final features = await data.features;
  final labels = await data.labels;

  final validator = Float32x4CrossValidator.kFold(numberOfFolds: 7);

  final step = 0.001;
  final limit = 0.6;

  double minError = double.infinity;
  double bestLearningRate = 0.0;

  for (double rate = step; rate < limit; rate += step) {
    final logisticRegressor = LogisticRegressor(
        iterationLimit: 100,
        learningRate: rate,
        batchSize: 1,
        learningRateType: LearningRateType.constant,
        fitIntercept: true);
    final error = validator.evaluate(logisticRegressor, features, labels, MetricType.accuracy);
    if (error < minError) {
      minError = error;
      bestLearningRate = rate;
    }
  }

  print('Best learning rate: $bestLearningRate');

  return minError;
}
