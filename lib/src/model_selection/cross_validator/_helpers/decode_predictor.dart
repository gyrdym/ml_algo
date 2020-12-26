import 'dart:convert';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor.dart';
import 'package:ml_algo/src/model_selection/cross_validator/predictor_type.dart';
import 'package:ml_algo/src/predictor/predictor.dart';

Predictor decodePredictor(
    PredictorType predictorType,
    Map<String, dynamic> json,
) {
  switch (predictorType) {
    case PredictorType.softmaxRegressor:
      return SoftmaxRegressor.fromJson(jsonEncode(json));

    case PredictorType.logisticRegressor:
      return LogisticRegressor.fromJson(jsonEncode(json));

    case PredictorType.knnClassifier:
      return KnnClassifier.fromJson(jsonEncode(json));

    case PredictorType.decisionTreeClassifier:
      return DecisionTreeClassifier.fromJson(jsonEncode(json));

    case PredictorType.knnRegressor:
      return KnnRegressor.fromJson(jsonEncode(json));

    case PredictorType.linearRegressor:
      return LinearRegressor.fromJson(jsonEncode(json));

    default:
      throw UnsupportedError('Unsupported predictor type - ${predictorType}');
  }
}
