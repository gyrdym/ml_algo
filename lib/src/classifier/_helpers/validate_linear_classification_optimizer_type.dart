import 'package:ml_algo/src/classifier/_constants/supported_linear_optimizer_types.dart';
import 'package:ml_algo/src/common/exception/unsupported_linear_optimizer_type_exception.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';

void validateLinearClassificationOptimizerType(LinearOptimizerType optimizerType) {
  if (![LinearOptimizerType.gradient, LinearOptimizerType.coordinate].contains(optimizerType)) {
    throw UnsupportedLinearOptimizerTypeException(optimizerType, supportedLinearOptimizerTypes);
  }
}
