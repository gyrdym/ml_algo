import 'package:ml_algo/src/model_selection/cross_validator/predictor_type.dart';
import 'package:ml_algo/src/model_selection/cross_validator/predictor_type_encoded_values.dart';

PredictorType decodePredictorType(String json) {
  switch (json) {
    case knnRegressorPredictorTypeEncodedValue:
      return PredictorType.knnRegressor;

    case linearRegressorPredictorTypeEncodedValue:
      return PredictorType.linearRegressor;

    case decisionTreeClassifierPredictorTypeEncodedValue:
      return PredictorType.decisionTreeClassifier;

    case knnClassifierPredictorTypeEncodedValue:
      return PredictorType.knnClassifier;

    case logisticRegressorPredictorTypeEncodedValue:
      return PredictorType.logisticRegressor;

    case softmaxRegressorPredictorTypeEncodedValue:
      return PredictorType.softmaxRegressor;

    default:
      throw UnsupportedError('Unsupported predictor encoded value - $json');
  }
}
