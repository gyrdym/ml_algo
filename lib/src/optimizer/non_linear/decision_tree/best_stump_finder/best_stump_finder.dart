import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

abstract class BestStumpFinder {
  Map<DecisionTreeNode, Matrix> find(Matrix samples, ZRange outcomesColumnRange,
      Iterable<ZRange> featuresColumnRanges,
      [Map<ZRange, List<Vector>> rangeToNominalValues]);
}
