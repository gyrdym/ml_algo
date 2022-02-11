import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';

class UnsupportedLinearOptimizerTypeException implements Exception {
  UnsupportedLinearOptimizerTypeException(LinearOptimizerType optimizerType,
      Iterable<LinearOptimizerType> supportedTypes)
      : message =
            'Unsupported linear optimizer type - $optimizerType. Supported optimizer types: $supportedTypes';

  final String message;

  @override
  String toString() => message;
}
