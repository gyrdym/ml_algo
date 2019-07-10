import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

abstract class StumpFactory {
  Map<DecisionTreeNode, Matrix> create(Matrix samples,
      ZRange splittingColumnRange, ZRange outcomeColumnRange,
      [List<Vector> nominalValues]);
}
