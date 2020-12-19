import 'dart:convert';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_impl.dart';

KnnRegressor createKnnRegressorFromJson(String json) {
  if (json.isEmpty) {
    throw Exception('Provided JSON object is empty, please provide a proper '
        'JSON object');
  }

  final decodedJson = jsonDecode(json) as Map<String, dynamic>;

  return KnnRegressorImpl.fromJson(decodedJson);
}
