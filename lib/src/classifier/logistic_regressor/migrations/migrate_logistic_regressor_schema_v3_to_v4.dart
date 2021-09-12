import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_json_keys.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type_json_encoded_values.dart';

Map<String, dynamic> migrateLogisticRegressorSchemaV3toV4(
    Map<String, dynamic> json) {
  final migratedJson = Map<String, dynamic>.from(json);

  if (migratedJson[logisticRegressorDecayJsonKey] == null) {
    print('WARNING. LogisticRegressor decoding, decay is null. '
        'Setting it to 1');

    migratedJson[logisticRegressorDecayJsonKey] = 1;
  }

  if (migratedJson[logisticRegressorLearningRateTypeJsonKey] ==
      learningRateTypeToEncodedValue[LearningRateType.decreasingAdaptive]) {
    migratedJson[logisticRegressorLearningRateTypeJsonKey] =
        learningRateTypeToEncodedValue[LearningRateType.timeBased];
  }

  return migratedJson;
}
