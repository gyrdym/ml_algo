import 'package:ml_linalg/matrix.dart';

abstract class ScoreToProbMapper {
  /// Accepts a matrix of scores, returns a matrix of probabilities
  ///
  /// Score is a multiplication of a feature value and the corresponding weight
  /// (coefficient)
  Matrix map(Matrix scoresByClasses);
}
