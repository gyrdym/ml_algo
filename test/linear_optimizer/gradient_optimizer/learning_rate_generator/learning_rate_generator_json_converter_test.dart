import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type_json_converter.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type_json_encoded_values.dart';
import 'package:test/test.dart';

void main() {
  group('LearningRateTypeJsonConverter', () {
    test('should decode ${LearningRateType.decreasingAdaptive} value', () {
      expect(const LearningRateTypeJsonConverter()
          .fromJson(decreasingAdaptiveLearningRateTypeJsonEncodedValue),
          LearningRateType.decreasingAdaptive);
    });

    test('should decode ${LearningRateType.constant} value', () {
      expect(const LearningRateTypeJsonConverter()
          .fromJson(constantLearningRateTypeJsonEncodedValue),
          LearningRateType.constant);
    });

    test('should return null for unknown encoded value', () {
      expect(const LearningRateTypeJsonConverter()
          .fromJson('unknown_value'), null);
    });

    test('should encode ${LearningRateType.decreasingAdaptive} value', () {
      expect(const LearningRateTypeJsonConverter()
          .toJson(LearningRateType.decreasingAdaptive),
          decreasingAdaptiveLearningRateTypeJsonEncodedValue);
    });

    test('should encode ${LearningRateType.constant} value', () {
      expect(const LearningRateTypeJsonConverter()
          .toJson(LearningRateType.constant),
          constantLearningRateTypeJsonEncodedValue);
    });

    test('should return null for unknown learning rate type', () {
      expect(const LearningRateTypeJsonConverter().toJson(null), null);
    });
  });
}
