import 'package:ml_linalg/matrix.dart';

abstract class TreeAssessor {
  /// Returns error on the whole split subset
  double getAggregatedError(Iterable<Matrix> split, int targetIdx);

  /// Returns error on a single split subset
  double getError(Matrix samples, int targetIdx);
}
