import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type_json_encoded_values.dart';

class LinearOptimizerTypeJsonConverter
    implements JsonConverter<LinearOptimizerType, String> {
  const LinearOptimizerTypeJsonConverter();

  @override
  LinearOptimizerType fromJson(String json) {
    switch (json) {
      case gradientLinearOptimizerTypeEncodedValue:
        return LinearOptimizerType.gradient;

      case coordinateLinearOptimizerTypeEncodedValue:
        return LinearOptimizerType.coordinate;

      case closedFormLinearOptimizerTypeEncodedValue:
        return LinearOptimizerType.closedForm;
    }

    throw Exception('LinearOptimizerType, '
        'fromJson: unknown encoded value - $json');
  }

  @override
  String toJson(LinearOptimizerType type) {
    switch (type) {
      case LinearOptimizerType.gradient:
        return gradientLinearOptimizerTypeEncodedValue;

      case LinearOptimizerType.coordinate:
        return coordinateLinearOptimizerTypeEncodedValue;

      case LinearOptimizerType.closedForm:
        return closedFormLinearOptimizerTypeEncodedValue;
    }
  }
}
