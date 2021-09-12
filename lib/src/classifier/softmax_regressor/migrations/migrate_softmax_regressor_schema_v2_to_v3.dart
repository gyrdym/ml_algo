import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_json_keys.dart';
import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type_json_encoded_values.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type_json_encoded_values.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type_json_encoded_values.dart';

Map<String, dynamic> migrateSoftmaxRegressorSchemaV2toV3(
    Map<String, dynamic> json) {
  if (json[jsonSchemaVersionJsonKey] != null &&
      (json[jsonSchemaVersionJsonKey] as num) > 2) {
    return json;
  }

  final migratedJson = Map<String, dynamic>.from(json);

  if (migratedJson[softmaxRegressorOptimizerTypeJsonKey] == null) {
    print('WARNING. SoftmaxRegressor decoding, linear optimizer type is null. '
        'Setting it to LinearOptimizerType.gradient');

    migratedJson[softmaxRegressorOptimizerTypeJsonKey] =
        gradientLinearOptimizerTypeEncodedValue;
  }

  if (migratedJson[softmaxRegressorIterationsLimitJsonKey] == null) {
    print('WARNING. SoftmaxRegressor decoding, iterations limit is null. '
        'Setting it to 100');

    migratedJson[softmaxRegressorIterationsLimitJsonKey] = 100;
  }

  if (migratedJson[softmaxRegressorLearningRateTypeJsonKey] == null) {
    print('WARNING. SoftmaxRegressor decoding, learning rate type is null. '
        'Setting it to LearningRateType.constant');

    migratedJson[softmaxRegressorLearningRateTypeJsonKey] =
        learningRateTypeToEncodedValue[LearningRateType.constant];
  }

  if (migratedJson[softmaxRegressorInitialCoefsTypeJsonKey] == null) {
    print(
        'WARNING. SoftmaxRegressor decoding, initial coefficients type is null. '
        'Setting it to InitialCoefficientsType.zeroes');

    migratedJson[softmaxRegressorInitialCoefsTypeJsonKey] =
        zeroesInitialCoefficientsTypeJsonEncodedValue;
  }

  if (migratedJson[softmaxRegressorInitialLearningRateJsonKey] == null) {
    print('WARNING. SoftmaxRegressor decoding, initial learning rate is null. '
        'Setting it to 1e-3');

    migratedJson[softmaxRegressorInitialLearningRateJsonKey] = 1e-3;
  }

  if (migratedJson[softmaxRegressorMinCoefsUpdateJsonKey] == null) {
    print(
        'WARNING. SoftmaxRegressor decoding, min coefficients update is null. '
        'Setting it to 1e-12');

    migratedJson[softmaxRegressorMinCoefsUpdateJsonKey] = 1e-12;
  }

  if (migratedJson[softmaxRegressorLambdaJsonKey] == null) {
    print('WARNING. SoftmaxRegressor decoding, lambda is null. '
        'Setting it to 0');

    migratedJson[softmaxRegressorLambdaJsonKey] = 0;
  }

  if (migratedJson[softmaxRegressorBatchSizeJsonKey] == null) {
    print('WARNING. SoftmaxRegressor decoding, batch size is null. '
        'Setting it to 1');

    migratedJson[softmaxRegressorBatchSizeJsonKey] = 1;
  }

  if (migratedJson[softmaxRegressorFittingDataNormalizedFlagJsonKey] == null) {
    print(
        'WARNING. SoftmaxRegressor decoding, fitting data normalized flag is null. '
        'Setting it to false');

    migratedJson[softmaxRegressorFittingDataNormalizedFlagJsonKey] = false;
  }

  migratedJson[jsonSchemaVersionJsonKey] = 3;

  return migratedJson;
}
