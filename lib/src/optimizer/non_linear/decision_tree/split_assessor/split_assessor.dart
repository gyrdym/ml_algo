import 'package:ml_linalg/matrix.dart';
import 'package:xrange/zrange.dart';

abstract class SplitAssessor {
  /// Returns error on the whole split subset
  double getAggregatedError(Iterable<Matrix> splitObservations,
      ZRange outcomesRange);

  /// Returns error on a single split subset
  double getError(Matrix splitObservations, ZRange outcomesRange);
}
