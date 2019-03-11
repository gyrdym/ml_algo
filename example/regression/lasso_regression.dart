import 'dart:async';

import 'package:ml_algo/ml_algo.dart';
import 'package:tuple/tuple.dart';

Future main() async {
  final data = DataFrame.fromCsv('datasets/advertising.csv',
      columns: [const Tuple2(1, 4)], labelName: 'Sales');
  final features = await data.features;
  final labels = await data.labels;
  final model = LinearRegressor.lasso(iterationsLimit: 100, lambda: 46420.0);
  final validator = CrossValidator.kFold();
  final error = validator.evaluate(model, features, labels, MetricType.mape);

  print('coefficients: ${model.weights}');
  print('error: ${error.toStringAsFixed(2)}%');
}
