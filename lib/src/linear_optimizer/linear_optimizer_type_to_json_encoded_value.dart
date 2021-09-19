import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type_json_encoded_values.dart';
import 'package:quiver/collection.dart';

final linearOptimizerTypeToEncodedValue = BiMap()
  ..addAll({
    LinearOptimizerType.gradient: gradientLinearOptimizerTypeEncodedValue,
    LinearOptimizerType.coordinate: coordinateLinearOptimizerTypeEncodedValue,
  });
