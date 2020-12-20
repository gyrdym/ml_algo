import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type_json_encoded_values.dart';

class RegularizationTypeJsonConverter implements
    JsonConverter<RegularizationType, String> {

  const RegularizationTypeJsonConverter();

  @override
  RegularizationType fromJson(String json) {
    switch (json) {
      case l1RegularizationTypeJsonEncodedValue:
        return RegularizationType.L1;

      case l2RegularizationTypeJsonEncodedValue:
        return RegularizationType.L2;

      default:
        return null;
    }
  }

  @override
  String toJson(RegularizationType type) {
    switch (type) {
      case RegularizationType.L1:
        return l1RegularizationTypeJsonEncodedValue;

      case RegularizationType.L2:
        return l2RegularizationTypeJsonEncodedValue;

      default:
        return null;
    }
  }
}
