import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type_json_encoded_values.dart';

class LearningRateTypeJsonConverter implements
    JsonConverter<LearningRateType, String> {

  const LearningRateTypeJsonConverter();

  @override
  LearningRateType fromJson(String json) {
    switch (json) {
      case decreasingAdaptiveLearningRateTypeJsonEncodedValue:
        return LearningRateType.decreasingAdaptive;

      case constantLearningRateTypeJsonEncodedValue:
        return LearningRateType.constant;

      default:
        return null;
    }
  }

  @override
  String toJson(LearningRateType type) {
    switch (type) {
      case LearningRateType.decreasingAdaptive:
        return decreasingAdaptiveLearningRateTypeJsonEncodedValue;

      case LearningRateType.constant:
        return constantLearningRateTypeJsonEncodedValue;

      default:
        return null;
    }
  }
}
