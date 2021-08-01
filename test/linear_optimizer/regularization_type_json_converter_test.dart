import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type_json_converter_nullable.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type_json_encoded_values.dart';
import 'package:test/test.dart';

void main() {
  group('RegularizationTypeJsonConverter', () {
    test('should decode ${RegularizationType.L1} type', () {
      expect(
          const RegularizationTypeJsonConverterNullable()
              .fromJson(l1RegularizationTypeJsonEncodedValue),
          RegularizationType.L1);
    });

    test('should decode ${RegularizationType.L2} type', () {
      expect(
          const RegularizationTypeJsonConverterNullable()
              .fromJson(l2RegularizationTypeJsonEncodedValue),
          RegularizationType.L2);
    });

    test('should return null for unknown encoded value', () {
      expect(
          const RegularizationTypeJsonConverterNullable()
              .fromJson('unknown_value'),
          null);
    });

    test('should encode ${RegularizationType.L1} type', () {
      expect(
          const RegularizationTypeJsonConverterNullable()
              .toJson(RegularizationType.L1),
          l1RegularizationTypeJsonEncodedValue);
    });

    test('should encode ${RegularizationType.L2} type', () {
      expect(
          const RegularizationTypeJsonConverterNullable()
              .toJson(RegularizationType.L2),
          l2RegularizationTypeJsonEncodedValue);
    });

    test('should return null for unknown decoded value', () {
      expect(
          const RegularizationTypeJsonConverterNullable().toJson(null), null);
    });
  });
}
