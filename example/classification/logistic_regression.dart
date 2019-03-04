import 'dart:async';

import 'package:ml_algo/ml_algo.dart';

Future main() async {
  final data = MLData.fromCsvFile('datasets/pima_indians_diabetes_database.csv',
    labelIdx: 8,
    categoryNameToEncoder: {
      'class variable (0 or 1)': CategoricalDataEncoderType.oneHot,
    },
  );

  final features = await data.features;
  final labels = await data.labels;

  final validator = CrossValidator.kFold(numberOfFolds: 5);

  final logisticRegressor = LinearClassifier.logisticRegressor(
      initialLearningRate: 0.00001,
      iterationsLimit: 7000,
      learningRateType: LearningRateType.constant,
      randomSeed: 150);

  final accuracy = validator.evaluate(
      logisticRegressor, features, labels, MetricType.accuracy);

  print('Accuracy is ${accuracy.toStringAsFixed(2)}');
}
