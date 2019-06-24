import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

abstract class BestStumpFinder {
  Iterable<Matrix> find(Matrix observations, ZRange outcomesRange,
      Iterable<ZRange> featuresRanges,
      [Map<ZRange, List<Vector>> rangeToCategoricalValues]);
}
