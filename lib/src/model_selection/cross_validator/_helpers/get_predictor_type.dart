import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor.dart';
import 'package:ml_algo/src/model_selection/cross_validator/predictor_type.dart';
import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor.dart';

PredictorType getPredictorType(Predictor predictor) {
  if (predictor is LogisticRegressor) {
    return PredictorType.logisticRegressor;
  }

  if (predictor is SoftmaxRegressor) {
    return PredictorType.softmaxRegressor;
  }

  if (predictor is DecisionTreeClassifier) {
    return PredictorType.decisionTreeClassifier;
  }

  if (predictor is KnnClassifier) {
    return PredictorType.knnClassifier;
  }

  if (predictor is LinearRegressor) {
    return PredictorType.linearRegressor;
  }

  if (predictor is KnnRegressor) {
    return PredictorType.knnRegressor;
  }

  throw UnsupportedError(
      'Unsupported predictor type - ${predictor.runtimeType}');
}
