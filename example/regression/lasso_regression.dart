import 'dart:async';
import 'dart:typed_data';

import 'package:ml_algo/ml_algo.dart';
import 'package:tuple/tuple.dart';

Future main() async {
  final data = MLData.fromCsvFile('datasets/advertising.csv',
      columns: [const Tuple2<int, int>(1, 4)], labelIdx: 4, dtype: Float32x4);
  final features = await data.features;
  final labels = await data.labels;
  final model = LinearRegressor.lasso(iterationsLimit: 100, lambda: 46420.0);
  final validator = CrossValidator.kFold();
  final error = validator.evaluate(model, features, labels, MetricType.mape);

  print('coefficients: ${model.weights}');
  print('error: ${error.toStringAsFixed(2)}%');
}
