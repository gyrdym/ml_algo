import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_json_keys.dart';
import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/common/constants/default_parameters/coordinate_optimization.dart';
import 'package:ml_algo/src/common/constants/default_parameters/gradient_optimization.dart';
import 'package:ml_algo/src/common/constants/default_parameters/linear_optimization.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type_json_encoded_values.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type_to_json_encoded_value.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type_to_json_encoded_value.dart';

Map<String, dynamic> migrateSoftmaxRegressorSchemaV2toV3(
    Map<String, dynamic> json) {
  if (json[jsonSchemaVersionJsonKey] != null &&
      (json[jsonSchemaVersionJsonKey] as num) > 2) {
    return json;
  }

  final migratedJson = Map<String, dynamic>.from(json);

  if (migratedJson[softmaxRegressorOptimizerTypeJsonKey] == null) {
    print('WARNING. SoftmaxRegressor decoding, linear optimizer type is null. '
        'Setting it to $linearOptimizerTypeDefaultValue');

    migratedJson[softmaxRegressorOptimizerTypeJsonKey] =
        linearOptimizerTypeToEncodedValue[linearOptimizerTypeDefaultValue];
  }

  if (migratedJson[softmaxRegressorIterationsLimitJsonKey] == null) {
    print('WARNING. SoftmaxRegressor decoding, iterations limit is null. '
        'Setting it to $iterationLimitDefaultValue');

    migratedJson[softmaxRegressorIterationsLimitJsonKey] =
        iterationLimitDefaultValue;
  }

  if (migratedJson[softmaxRegressorLearningRateTypeJsonKey] == null) {
    print('WARNING. SoftmaxRegressor decoding, learning rate type is null. '
        'Setting it to $learningRateTypeDefaultValue');

    migratedJson[softmaxRegressorLearningRateTypeJsonKey] =
        learningRateTypeToEncodedValue[learningRateTypeDefaultValue];
  }

  if (migratedJson[softmaxRegressorInitialCoefsTypeJsonKey] == null) {
    print(
        'WARNING. SoftmaxRegressor decoding, initial coefficients type is null. '
        'Setting it to $initialCoefficientsTypeDefaultValue');

    migratedJson[softmaxRegressorInitialCoefsTypeJsonKey] =
        initialCoefficientsTypeToEncodedValue[
            initialCoefficientsTypeDefaultValue];
  }

  if (migratedJson[softmaxRegressorInitialLearningRateJsonKey] == null) {
    print('WARNING. SoftmaxRegressor decoding, initial learning rate is null. '
        'Setting it to $initialLearningRateDefaultValue');

    migratedJson[softmaxRegressorInitialLearningRateJsonKey] =
        initialLearningRateDefaultValue;
  }

  if (migratedJson[softmaxRegressorMinCoefsUpdateJsonKey] == null) {
    print(
        'WARNING. SoftmaxRegressor decoding, min coefficients update is null. '
        'Setting it to $minCoefficientsUpdateDefaultValue');

    migratedJson[softmaxRegressorMinCoefsUpdateJsonKey] =
        minCoefficientsUpdateDefaultValue;
  }

  if (migratedJson[softmaxRegressorLambdaJsonKey] == null) {
    print('WARNING. SoftmaxRegressor decoding, lambda is null. '
        'Setting it to $lambdaDefaultValue');

    migratedJson[softmaxRegressorLambdaJsonKey] = lambdaDefaultValue;
  }

  if (migratedJson[softmaxRegressorBatchSizeJsonKey] == null) {
    print('WARNING. SoftmaxRegressor decoding, batch size is null. '
        'Setting it to $batchSizeDefaultValue');

    migratedJson[softmaxRegressorBatchSizeJsonKey] = batchSizeDefaultValue;
  }

  if (migratedJson[softmaxRegressorFittingDataNormalizedFlagJsonKey] == null) {
    print(
        'WARNING. SoftmaxRegressor decoding, fitting data normalized flag is null. '
        'Setting it to $isFittingDataNormalizedDefaultValue');

    migratedJson[softmaxRegressorFittingDataNormalizedFlagJsonKey] =
        isFittingDataNormalizedDefaultValue;
  }

  migratedJson[jsonSchemaVersionJsonKey] = 3;

  return migratedJson;
}
