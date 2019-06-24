import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

abstract class StumpSelector {
  Iterable<Matrix> select(Matrix observations, ZRange splittingFeatureRange,
      ZRange outcomesRange, [List<Vector> categoricalValues]);
}
