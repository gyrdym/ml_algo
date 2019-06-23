import 'package:ml_linalg/matrix.dart';
import 'package:xrange/zrange.dart';

abstract class LeafDetector {
  bool isLeaf(Matrix observations, ZRange outcomesRange, int nodesCount);
}
