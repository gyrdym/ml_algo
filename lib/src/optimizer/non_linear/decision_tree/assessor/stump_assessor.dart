import 'package:ml_linalg/matrix.dart';
import 'package:xrange/zrange.dart';

abstract class StumpAssessor {
  double getErrorOnStump(Iterable<Matrix> stumpObservations,
      ZRange outcomesRange);
  double getErrorOnNode(Matrix nodeObservations, ZRange outcomesRange);
}
