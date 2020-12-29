import 'package:ml_algo/src/model_selection/cross_validator/predictor_type.dart';
import 'package:ml_algo/src/model_selection/cross_validator/predictor_type_encoded_values.dart';

String encodePredictorType(PredictorType predictorType) {
  switch (predictorType) {
    case PredictorType.knnRegressor:
      return knnRegressorPredictorTypeEncodedValue;

    case PredictorType.linearRegressor:
      return linearRegressorPredictorTypeEncodedValue;

    case PredictorType.decisionTreeClassifier:
      return decisionTreeClassifierPredictorTypeEncodedValue;

    case PredictorType.knnClassifier:
      return knnClassifierPredictorTypeEncodedValue;

    case PredictorType.logisticRegressor:
      return logisticRegressorPredictorTypeEncodedValue;

    case PredictorType.softmaxRegressor:
      return softmaxRegressorPredictorTypeEncodedValue;

    default:
      throw UnsupportedError('Unsupported predictor type - $predictorType');
  }
}
