import 'dart:async';

import 'package:ml_algo/float32x4_cross_validator.dart';
import 'package:ml_algo/float32x4_csv_ml_data.dart';
import 'package:ml_algo/lasso_regressor.dart';
import 'package:ml_algo/metric_type.dart';

Future main() async {
  final data = Float32x4CsvMLData.fromFile('datasets/advertising.csv', labelIdx: 4);
  final features = await data.features;
  final labels = await data.labels;
  final model = LassoRegressor(iterationLimit: 100, lambda: 6750.0);
  final validator = Float32x4CrossValidator.kFold();
  final error = validator.evaluate(model, features, labels, MetricType.mape);

  print('coefficients: ${model.weights}');
  print('error: ${error.toStringAsFixed(2)}%');
}
