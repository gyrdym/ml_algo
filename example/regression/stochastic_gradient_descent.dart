import 'dart:async';

import 'package:ml_algo/ml_algo.dart';

Future bostonHousingRegression() async {
  final data = DataFrame.fromCsv('datasets/housing.csv',
    headerExists: false,
    fieldDelimiter: ' ',
    labelIdx: 13,
  );

  final features = (await data.features)
      .mapColumns((column) => column.normalize());
  final labels = await data.labels;

  final folds = 5;
  final validator = CrossValidator.kFold(numberOfFolds: folds);

  final regressor = LinearRegressor.gradient(
      gradientType: GradientType.stochastic,
      iterationsLimit: 100,
      initialLearningRate: 5.0,
      minWeightsUpdate: 1e-4,
      randomSeed: 20,
      learningRateType: LearningRateType.constant);

  final error =
    validator.evaluate(regressor, features, labels, MetricType.mape);

  print('Linear regression on Boston housing dataset, label - `medv`, MAPE '
      'error on k-fold validation ($folds folds): '
      '${error.toStringAsFixed(2)}%');
}

Future main() async {
  await bostonHousingRegression();
}
