import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_json_keys.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type_json_encoded_values.dart';

Map<String, dynamic> migrateSoftmaxRegressorSchemaV3toV4(
    Map<String, dynamic> json) {
  final migratedJson = Map<String, dynamic>.from(json);

  if (migratedJson[softmaxRegressorDecayJsonKey] == null) {
    print('WARNING. SoftmaxRegressor decoding, decay is null. '
        'Setting it to 1');

    migratedJson[softmaxRegressorDecayJsonKey] = 1;
  }

  if (migratedJson[softmaxRegressorDropRateJsonKey] == null) {
    print('WARNING. SoftmaxRegressor decoding, dropRate is null. '
        'Setting it to 10');

    migratedJson[softmaxRegressorDropRateJsonKey] = 10;
  }

  if (migratedJson[softmaxRegressorLearningRateTypeJsonKey] ==
      learningRateTypeToEncodedValue[LearningRateType.decreasingAdaptive]) {
    migratedJson[softmaxRegressorLearningRateTypeJsonKey] =
        learningRateTypeToEncodedValue[LearningRateType.timeBased];
  }

  return migratedJson;
}
