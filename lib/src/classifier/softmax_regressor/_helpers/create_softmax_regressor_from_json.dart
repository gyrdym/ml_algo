import 'dart:convert';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_impl.dart';

SoftmaxRegressor createSoftmaxRegressorFromJson(String json) {
  final decoded = jsonDecode(json) as Map<String, dynamic>;
  return SoftmaxRegressorImpl.fromJson(decoded);
}
