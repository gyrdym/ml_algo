import 'dart:convert';

import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_impl.dart';

LinearRegressor createLinearRegressorFromJson(String json) {
  final decodedJson = jsonDecode(json) as Map<String, dynamic>;

  return LinearRegressorImpl.fromJson(decodedJson);
}
