import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type_json_encoded_values.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_json_keys.dart';

Map<String, dynamic> migrateLinearRegressorSchemaV2toV3(
    Map<String, dynamic> json) {
  final migratedJson = Map<String, dynamic>.from(json);

  if (migratedJson[linearRegressorDecayJsonKey] == null) {
    print('WARNING. LinearRegressor decoding, decay is null. '
        'Setting it to 1');

    migratedJson[linearRegressorDecayJsonKey] = 1;
  }

  if (migratedJson[linearRegressorDropRateJsonKey] == null) {
    print('WARNING. LinearRegressor decoding, dropRate is null. '
        'Setting it to 10');

    migratedJson[linearRegressorDropRateJsonKey] = 10;
  }

  if (migratedJson[linearRegressorLearningRateTypeJsonKey] ==
      learningRateTypeToEncodedValue[LearningRateType.decreasingAdaptive]) {
    migratedJson[linearRegressorLearningRateTypeJsonKey] =
        learningRateTypeToEncodedValue[LearningRateType.timeBased];
  }

  return migratedJson;
}
