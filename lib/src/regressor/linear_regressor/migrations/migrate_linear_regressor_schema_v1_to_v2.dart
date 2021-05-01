import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type_json_encoded_values.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type_json_encoded_values.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type_json_encoded_values.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_json_keys.dart';

Map<String, dynamic> migrateLinearRegressorSchemaV1toV2(
    Map<String, dynamic> json) {
  if (json[jsonSchemaVersionJsonKey] != null &&
      (json[jsonSchemaVersionJsonKey] as num) > 1) {
    return json;
  }

  final migratedJson = Map<String, dynamic>.from(json);

  if (migratedJson[linearRegressorOptimizerTypeJsonKey] == null) {
    print('WARNING. LinearRegressor decoding, linear optimizer type is null. '
        'Setting it to LinearOptimizerType.gradient');

    migratedJson[linearRegressorOptimizerTypeJsonKey] =
        gradientLinearOptimizerTypeEncodedValue;
  }

  if (migratedJson[linearRegressorIterationsLimitJsonKey] == null) {
    print('WARNING. LinearRegressor decoding, iterations limit is null. '
        'Setting it to 100');

    migratedJson[linearRegressorIterationsLimitJsonKey] = 100;
  }

  if (migratedJson[linearRegressorLearningRateTypeJsonKey] == null) {
    print('WARNING. LinearRegressor decoding, learning rate type is null. '
        'Setting it to LearningRateType.constant');

    migratedJson[linearRegressorLearningRateTypeJsonKey] =
        learningRateTypeToEncodedValue[LearningRateType.constant];
  }

  if (migratedJson[linearRegressorInitialCoefficientsTypeJsonKey] == null) {
    print(
        'WARNING. LinearRegressor decoding, initial coefficients type is null. '
        'Setting it to InitialCoefficientsType.zeroes');

    migratedJson[linearRegressorInitialCoefficientsTypeJsonKey] =
        zeroesInitialCoefficientsTypeJsonEncodedValue;
  }

  if (migratedJson[linearRegressorInitialLearningRateJsonKey] == null) {
    print('WARNING. LinearRegressor decoding, initial learning rate is null. '
        'Setting it to 1e-3');

    migratedJson[linearRegressorInitialLearningRateJsonKey] = 1e-3;
  }

  if (migratedJson[linearRegressorMinCoefficientsUpdateJsonKey] == null) {
    print('WARNING. LinearRegressor decoding, min coefficients update is null. '
        'Setting it to 1e-12');

    migratedJson[linearRegressorMinCoefficientsUpdateJsonKey] = 1e-12;
  }

  if (migratedJson[linearRegressorLambdaJsonKey] == null) {
    print('WARNING. LinearRegressor decoding, lambda is null. '
        'Setting it to 0');

    migratedJson[linearRegressorLambdaJsonKey] = 0;
  }

  if (migratedJson[linearRegressorBatchSizeJsonKey] == null) {
    print('WARNING. LinearRegressor decoding, batch size is null. '
        'Setting it to 1');

    migratedJson[linearRegressorBatchSizeJsonKey] = 1;
  }

  if (migratedJson[linearRegressorFittingDataNormalizedFlagJsonKey] == null) {
    print(
        'WARNING. LinearRegressor decoding, fitting data normalized flag is null. '
        'Setting it to false');

    migratedJson[linearRegressorFittingDataNormalizedFlagJsonKey] = false;
  }

  migratedJson[jsonSchemaVersionJsonKey] = 2;

  return migratedJson;
}
