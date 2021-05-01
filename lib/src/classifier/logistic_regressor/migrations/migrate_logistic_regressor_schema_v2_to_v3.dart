import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_json_keys.dart';
import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type_json_encoded_values.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type_json_encoded_values.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type_json_encoded_values.dart';

Map<String, dynamic> migrateLogisticRegressorSchemaV2toV3(
    Map<String, dynamic> json) {
  if (json[jsonSchemaVersionJsonKey] != null &&
      (json[jsonSchemaVersionJsonKey] as num) > 2) {
    return json;
  }

  final migratedJson = Map<String, dynamic>.from(json);

  if (migratedJson[logisticRegressorLinearOptimizerTypeJsonKey] == null) {
    print('WARNING. LogisticRegressor decoding, linear optimizer type is null. '
        'Setting it to LinearOptimizerType.gradient');

    migratedJson[logisticRegressorLinearOptimizerTypeJsonKey] =
        gradientLinearOptimizerTypeEncodedValue;
  }

  if (migratedJson[logisticRegressorIterationsLimitJsonKey] == null) {
    print('WARNING. LogisticRegressor decoding, iterations limit is null. '
        'Setting it to 100');

    migratedJson[logisticRegressorIterationsLimitJsonKey] = 100;
  }

  if (migratedJson[logisticRegressorLearningRateTypeJsonKey] == null) {
    print('WARNING. LogisticRegressor decoding, learning rate type is null. '
        'Setting it to LearningRateType.constant');

    migratedJson[logisticRegressorLearningRateTypeJsonKey] =
        learningRateTypeToEncodedValue[LearningRateType.constant];
  }

  if (migratedJson[logisticRegressorInitCoefficientsTypeJsonKey] == null) {
    print(
        'WARNING. LogisticRegressor decoding, initial coefficients type is null. '
        'Setting it to InitialCoefficientsType.zeroes');

    migratedJson[logisticRegressorInitCoefficientsTypeJsonKey] =
        zeroesInitialCoefficientsTypeJsonEncodedValue;
  }

  if (migratedJson[logisticRegressorInitialLearningRateJsonKey] == null) {
    print('WARNING. LogisticRegressor decoding, initial learning rate is null. '
        'Setting it to 1e-3');

    migratedJson[logisticRegressorInitialLearningRateJsonKey] = 1e-3;
  }

  if (migratedJson[logisticRegressorMinCoefsUpdateJsonKey] == null) {
    print(
        'WARNING. LogisticRegressor decoding, min coefficients update is null. '
        'Setting it to 1e-12');

    migratedJson[logisticRegressorMinCoefsUpdateJsonKey] = 1e-12;
  }

  if (migratedJson[logisticRegressorLambdaJsonKey] == null) {
    print('WARNING. LogisticRegressor decoding, lambda is null. '
        'Setting it to 0');

    migratedJson[logisticRegressorLambdaJsonKey] = 0;
  }

  if (migratedJson[logisticRegressorBatchSizeJsonKey] == null) {
    print('WARNING. LogisticRegressor decoding, batch size is null. '
        'Setting it to 1');

    migratedJson[logisticRegressorBatchSizeJsonKey] = 1;
  }

  if (migratedJson[logisticRegressorDataNormalizedFlagJsonKey] == null) {
    print(
        'WARNING. LogisticRegressor decoding, fitting data normalized flag is null. '
        'Setting it to false');

    migratedJson[logisticRegressorDataNormalizedFlagJsonKey] = false;
  }

  migratedJson[jsonSchemaVersionJsonKey] = 3;

  return migratedJson;
}
