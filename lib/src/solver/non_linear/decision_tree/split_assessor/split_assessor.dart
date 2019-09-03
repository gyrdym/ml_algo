import 'package:ml_linalg/matrix.dart';

abstract class SplitAssessor {
  /// Returns error on the whole split subset
  double getAggregatedError(Iterable<Matrix> splitObservations,
      int targetId);

  /// Returns error on a single split subset
  double getError(Matrix splitObservations, int targetId);
}
