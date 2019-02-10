import 'dart:async';
import 'dart:typed_data';

import 'package:ml_algo/cross_validator.dart';
import 'package:ml_algo/linear_regressor.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/ml_data.dart';

Future main() async {
  final data = MLData.fromCsvFile('datasets/advertising.csv', labelIdx: 4, dtype: Float32x4);
  final features = await data.features;
  final labels = await data.labels;
  final model = LinearRegressor.lasso(iterationsLimit: 100, lambda: 6750.0);
  final validator = CrossValidator.kFold();
  final error = validator.evaluate(model, features, labels, MetricType.mape);

  print('coefficients: ${model.weights}');
  print('error: ${error.toStringAsFixed(2)}%');
}
