import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type_json_encoded_values.dart';

class LearningRateTypeJsonConverter implements
    JsonConverter<LearningRateType, String> {

  const LearningRateTypeJsonConverter();

  @override
  LearningRateType fromJson(String json) =>
      learningRateTypeToEncodedValue.inverse[json] ?? defaultLearningRateType;

  @override
  String toJson(LearningRateType type) =>
      learningRateTypeToEncodedValue[type]
          ?? learningRateTypeToEncodedValue[defaultLearningRateType]!;
}
