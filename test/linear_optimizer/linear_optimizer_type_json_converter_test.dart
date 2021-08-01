import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type_json_converter.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type_json_encoded_values.dart';
import 'package:test/test.dart';

void main() {
  group('LinearOptimizerTypeJsonConverter', () {
    test('should decode ${LinearOptimizerType.gradient} type', () {
      expect(
          const LinearOptimizerTypeJsonConverter()
              .fromJson(gradientLinearOptimizerTypeEncodedValue),
          LinearOptimizerType.gradient);
    });

    test('should decode ${LinearOptimizerType.coordinate} type', () {
      expect(
          const LinearOptimizerTypeJsonConverter()
              .fromJson(coordinateLinearOptimizerTypeEncodedValue),
          LinearOptimizerType.coordinate);
    });

    test('should throw exception if unknown value is provided', () {
      expect(
        () =>
            const LinearOptimizerTypeJsonConverter().fromJson('unknown_value'),
        throwsException,
      );
    });

    test('should encode ${LinearOptimizerType.gradient} type', () {
      expect(
          const LinearOptimizerTypeJsonConverter()
              .toJson(LinearOptimizerType.gradient),
          gradientLinearOptimizerTypeEncodedValue);
    });

    test('should encode ${LinearOptimizerType.coordinate} type', () {
      expect(
          const LinearOptimizerTypeJsonConverter()
              .toJson(LinearOptimizerType.coordinate),
          coordinateLinearOptimizerTypeEncodedValue);
    });
  });
}
