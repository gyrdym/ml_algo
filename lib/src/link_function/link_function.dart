import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_linalg/matrix.dart';

abstract class LinkFunction {
  /// Accepts a matrix of scores, returns a matrix of probabilities
  ///
  /// Score is a multiplication of a feature value and the corresponding weight
  /// (coefficient)
  Matrix link(Matrix scoresByClasses);
}
