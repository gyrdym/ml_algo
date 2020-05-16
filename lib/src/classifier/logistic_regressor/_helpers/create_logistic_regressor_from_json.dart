import 'dart:convert';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_impl.dart';

LogisticRegressor createLogisticRegressorFromJson(String json) {
  final decoded = jsonDecode(json) as Map<String, dynamic>;
  return LogisticRegressorImpl.fromJson(decoded);
}
