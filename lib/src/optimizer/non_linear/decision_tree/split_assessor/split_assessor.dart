import 'package:ml_linalg/matrix.dart';
import 'package:xrange/zrange.dart';

abstract class SplitAssessor {
  double getAggregatedError(Iterable<Matrix> splitObservations,
      ZRange outcomesRange);
  double getError(Matrix splitObservations, ZRange outcomesRange);
}
