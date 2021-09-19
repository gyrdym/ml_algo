import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_json_keys.dart';
import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/common/constants/default_parameters/coordinate_optimization.dart';
import 'package:ml_algo/src/common/constants/default_parameters/gradient_optimization.dart';
import 'package:ml_algo/src/common/constants/default_parameters/linear_optimization.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type_json_encoded_values.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type_to_json_encoded_value.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type_to_json_encoded_value.dart';

Map<String, dynamic> migrateLogisticRegressorSchemaV2toV3(
    Map<String, dynamic> json) {
  if (json[jsonSchemaVersionJsonKey] != null &&
      (json[jsonSchemaVersionJsonKey] as num) > 2) {
    return json;
  }

  final migratedJson = Map<String, dynamic>.from(json);

  if (migratedJson[logisticRegressorLinearOptimizerTypeJsonKey] == null) {
    print('WARNING. LogisticRegressor decoding, linear optimizer type is null. '
        'Setting it to $linearOptimizerTypeDefaultValue');

    migratedJson[logisticRegressorLinearOptimizerTypeJsonKey] =
        linearOptimizerTypeToEncodedValue[linearOptimizerTypeDefaultValue];
  }

  if (migratedJson[logisticRegressorIterationsLimitJsonKey] == null) {
    print('WARNING. LogisticRegressor decoding, iterations limit is null. '
        'Setting it to $iterationLimitDefaultValue');

    migratedJson[logisticRegressorIterationsLimitJsonKey] =
        iterationLimitDefaultValue;
  }

  if (migratedJson[logisticRegressorLearningRateTypeJsonKey] == null) {
    print('WARNING. LogisticRegressor decoding, learning rate type is null. '
        'Setting it to $learningRateTypeDefaultValue');

    migratedJson[logisticRegressorLearningRateTypeJsonKey] =
        learningRateTypeToEncodedValue[learningRateTypeDefaultValue];
  }

  if (migratedJson[logisticRegressorInitCoefficientsTypeJsonKey] == null) {
    print(
        'WARNING. LogisticRegressor decoding, initial coefficients type is null. '
        'Setting it to $initialCoefficientsTypeDefaultValue');

    migratedJson[logisticRegressorInitCoefficientsTypeJsonKey] =
        initialCoefficientsTypeToEncodedValue[
            initialCoefficientsTypeDefaultValue];
  }

  if (migratedJson[logisticRegressorInitialLearningRateJsonKey] == null) {
    print('WARNING. LogisticRegressor decoding, initial learning rate is null. '
        'Setting it to $initialLearningRateDefaultValue');

    migratedJson[logisticRegressorInitialLearningRateJsonKey] =
        initialLearningRateDefaultValue;
  }

  if (migratedJson[logisticRegressorMinCoefsUpdateJsonKey] == null) {
    print(
        'WARNING. LogisticRegressor decoding, min coefficients update is null. '
        'Setting it to $minCoefficientsUpdateDefaultValue');

    migratedJson[logisticRegressorMinCoefsUpdateJsonKey] =
        minCoefficientsUpdateDefaultValue;
  }

  if (migratedJson[logisticRegressorLambdaJsonKey] == null) {
    print('WARNING. LogisticRegressor decoding, lambda is null. '
        'Setting it to $lambdaDefaultValue');

    migratedJson[logisticRegressorLambdaJsonKey] = lambdaDefaultValue;
  }

  if (migratedJson[logisticRegressorBatchSizeJsonKey] == null) {
    print('WARNING. LogisticRegressor decoding, batch size is null. '
        'Setting it to $batchSizeDefaultValue');

    migratedJson[logisticRegressorBatchSizeJsonKey] = batchSizeDefaultValue;
  }

  if (migratedJson[logisticRegressorDataNormalizedFlagJsonKey] == null) {
    print(
        'WARNING. LogisticRegressor decoding, fitting data normalized flag is null. '
        'Setting it to $isFittingDataNormalizedDefaultValue');

    migratedJson[logisticRegressorDataNormalizedFlagJsonKey] =
        isFittingDataNormalizedDefaultValue;
  }

  migratedJson[jsonSchemaVersionJsonKey] = 3;

  return migratedJson;
}
