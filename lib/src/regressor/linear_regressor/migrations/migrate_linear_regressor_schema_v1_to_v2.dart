import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/common/constants/default_parameters/common.dart';
import 'package:ml_algo/src/common/constants/default_parameters/coordinate_optimization.dart';
import 'package:ml_algo/src/common/constants/default_parameters/gradient_optimization.dart';
import 'package:ml_algo/src/common/constants/default_parameters/linear_optimization.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type_json_encoded_values.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type_to_json_encoded_value.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type_to_json_encoded_value.dart';
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
        'Setting it to $linearOptimizerTypeDefaultValue');

    migratedJson[linearRegressorOptimizerTypeJsonKey] =
        linearOptimizerTypeToEncodedValue[linearOptimizerTypeDefaultValue];
  }

  if (migratedJson[linearRegressorIterationsLimitJsonKey] == null) {
    print('WARNING. LinearRegressor decoding, iterations limit is null. '
        'Setting it to $iterationLimitDefaultValue');

    migratedJson[linearRegressorIterationsLimitJsonKey] =
        iterationLimitDefaultValue;
  }

  if (migratedJson[linearRegressorLearningRateTypeJsonKey] == null) {
    print('WARNING. LinearRegressor decoding, learning rate type is null. '
        'Setting it to $learningRateTypeDefaultValue');

    migratedJson[linearRegressorLearningRateTypeJsonKey] =
        learningRateTypeToEncodedValue[learningRateTypeDefaultValue];
  }

  if (migratedJson[linearRegressorInitialCoefficientsTypeJsonKey] == null) {
    print(
        'WARNING. LinearRegressor decoding, initial coefficients type is null. '
        'Setting it to $initialCoefficientsTypeDefaultValue');

    migratedJson[linearRegressorInitialCoefficientsTypeJsonKey] =
        initialCoefficientsTypeToEncodedValue[
            initialCoefficientsTypeDefaultValue];
  }

  if (migratedJson[linearRegressorInitialLearningRateJsonKey] == null) {
    print('WARNING. LinearRegressor decoding, initial learning rate is null. '
        'Setting it to $initialLearningRateDefaultValue');

    migratedJson[linearRegressorInitialLearningRateJsonKey] =
        initialLearningRateDefaultValue;
  }

  if (migratedJson[linearRegressorMinCoefficientsUpdateJsonKey] == null) {
    print('WARNING. LinearRegressor decoding, min coefficients update is null. '
        'Setting it to $minCoefficientsUpdateDefaultValue');

    migratedJson[linearRegressorMinCoefficientsUpdateJsonKey] =
        minCoefficientsUpdateDefaultValue;
  }

  if (migratedJson[linearRegressorLambdaJsonKey] == null) {
    print('WARNING. LinearRegressor decoding, lambda is null. '
        'Setting it to $lambdaDefaultValue');

    migratedJson[linearRegressorLambdaJsonKey] = lambdaDefaultValue;
  }

  if (migratedJson[linearRegressorBatchSizeJsonKey] == null) {
    print('WARNING. LinearRegressor decoding, batch size is null. '
        'Setting it to $batchSizeDefaultValue');

    migratedJson[linearRegressorBatchSizeJsonKey] = batchSizeDefaultValue;
  }

  if (migratedJson[linearRegressorFittingDataNormalizedFlagJsonKey] == null) {
    print(
        'WARNING. LinearRegressor decoding, fitting data normalized flag is null. '
        'Setting it to $isFittingDataNormalizedDefaultValue');

    migratedJson[linearRegressorFittingDataNormalizedFlagJsonKey] =
        isFittingDataNormalizedDefaultValue;
  }

  migratedJson[jsonSchemaVersionJsonKey] = 2;

  return migratedJson;
}
