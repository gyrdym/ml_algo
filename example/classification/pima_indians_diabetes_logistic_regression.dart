import 'dart:async';

import 'package:ml_algo/float32x4_cross_validator.dart';
import 'package:ml_algo/float32x4_csv_ml_data.dart';
import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/logistic_regressor.dart';
import 'package:ml_algo/metric_type.dart';

Future main() async {
  final data = Float32x4CsvMLData.fromFile('datasets/pima_indians_diabetes_database.csv', labelIdx: 8);

  final features = await data.features;
  final labels = await data.labels;

  final validator = Float32x4CrossValidator.kFold(numberOfFolds: 5);

  final step = 0.00001;
  final start = 0.001;
  final limit = 0.01;

  double minError = double.infinity;
  double bestLearningRate = 0.0;

  // Let's find optimal learningRate. WARNING: it may take very much time!
  for (double rate = start; rate < limit; rate += step) {
    final logisticRegressor = LogisticRegressor(
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
