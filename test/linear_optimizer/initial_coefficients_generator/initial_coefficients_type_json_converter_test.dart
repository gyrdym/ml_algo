import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type_json_converter.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type_json_encoded_values.dart';
import 'package:test/test.dart';

void main() {
  group('InitialCoefficientsTypeJsonConverter', () {
    test('should decode ${InitialCoefficientsType.zeroes} type', () {
      expect(
          const InitialCoefficientsTypeJsonConverter()
              .fromJson(zeroesInitialCoefficientsTypeJsonEncodedValue),
          InitialCoefficientsType.zeroes);
    });

    test('should throw exception in case of unknown encoded value', () {
      expect(
          () => const InitialCoefficientsTypeJsonConverter()
              .fromJson('unknown_value'),
          throwsException);
    });

    test('should encode ${InitialCoefficientsType.zeroes} type', () {
      expect(
          const InitialCoefficientsTypeJsonConverter()
              .toJson(InitialCoefficientsType.zeroes),
          zeroesInitialCoefficientsTypeJsonEncodedValue);
    });
  });
}
