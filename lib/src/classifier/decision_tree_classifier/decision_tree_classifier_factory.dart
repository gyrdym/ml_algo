import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_linalg/dtype.dart';

abstract class DecisionTreeClassifierFactory {
  DecisionTreeClassifier create(
      num minError,
      int minSamplesCount,
      int maxDepth,
      TreeNode root,
      String targetName,
      DType dtype,
  );
}
