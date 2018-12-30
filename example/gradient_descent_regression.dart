import 'dart:async';

import 'package:ml_algo/ml_algo.dart';

Future<double> gradientDescentRegression() async {
  final data = Float32x4CsvMLData.fromFile('datasets/advertising.csv', 4);
  final features = await data.features;
  final labels = await data.labels;

  final sgdRegressionModel = GradientRegressor(
      type: GradientType.stochastic,
      iterationLimit: 100000,
      learningRate: 1e-5,
      learningRateType: LearningRateType.constant);

  final validator = Float32x4CrossValidator.kFold();
  final quality = validator.evaluate(sgdRegressionModel, features, labels, MetricType.mape);

  return quality;
}
