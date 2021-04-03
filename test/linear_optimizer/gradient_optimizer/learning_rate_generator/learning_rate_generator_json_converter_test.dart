import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type_json_converter.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type_json_encoded_values.dart';
import 'package:test/test.dart';

void main() {
  group('LearningRateTypeJsonConverter', () {
    test('should decode ${LearningRateType.decreasingAdaptive} value', () {
      expect(const LearningRateTypeJsonConverter()
          .fromJson(learningRateTypeToEncodedValue[LearningRateType.decreasingAdaptive]!),
          LearningRateType.decreasingAdaptive);
    });

    test('should decode ${LearningRateType.constant} value', () {
      expect(const LearningRateTypeJsonConverter()
          .fromJson(learningRateTypeToEncodedValue[LearningRateType.constant]!),
          LearningRateType.constant);
    });

    test('should return null for unknown encoded value', () {
      expect(const LearningRateTypeJsonConverter()
          .fromJson('unknown_value'), null);
    });

    test('should encode ${LearningRateType.decreasingAdaptive} value', () {
      expect(const LearningRateTypeJsonConverter()
          .toJson(LearningRateType.decreasingAdaptive),
          learningRateTypeToEncodedValue[LearningRateType.decreasingAdaptive]!);
    });

    test('should encode ${LearningRateType.constant} value', () {
      expect(const LearningRateTypeJsonConverter()
          .toJson(LearningRateType.constant),
          learningRateTypeToEncodedValue[LearningRateType.constant]!);
    });
  });
}
