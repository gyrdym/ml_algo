import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type_json_encoded_values.dart';
import 'package:quiver/collection.dart';

final initialCoefficientsTypeToEncodedValue = BiMap()
  ..addAll({
    InitialCoefficientsType.zeroes:
        zeroesInitialCoefficientsTypeJsonEncodedValue,
  });
