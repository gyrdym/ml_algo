import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

abstract class ScoreToProbMapper {
  /// Accepts a vector of scores, returns a vector of probabilities
  /// Score is a multiplication of a feature value and the corresponding weight
  /// (coefficient)
  MLVector linkScoresToProbs(MLVector scores, [MLMatrix scoresByClasses]);
}
