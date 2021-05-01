import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type_json_encoded_values.dart';

class InitialCoefficientsTypeJsonConverter
    implements JsonConverter<InitialCoefficientsType, String> {
  const InitialCoefficientsTypeJsonConverter();

  @override
  InitialCoefficientsType fromJson(String json) {
    switch (json) {
      case zeroesInitialCoefficientsTypeJsonEncodedValue:
        return InitialCoefficientsType.zeroes;

      default:
        throw Exception('InitialCoefficientsTypeJsonConverter, fromJson: '
            'unknown encoded value - $json');
    }
  }

  @override
  String toJson(InitialCoefficientsType type) {
    switch (type) {
      case InitialCoefficientsType.zeroes:
        return zeroesInitialCoefficientsTypeJsonEncodedValue;
    }
  }
}
