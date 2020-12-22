import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_linalg/dtype.dart';

class DecisionTreeClassifierFactoryImpl implements
    DecisionTreeClassifierFactory {
  const DecisionTreeClassifierFactoryImpl();

  @override
  DecisionTreeClassifier create(
      num minError,
      int minSamplesCount,
      int maxDepth,
      TreeNode root,
      String targetName,
      DType dtype,
  ) => DecisionTreeClassifierImpl(
    minError,
    minSamplesCount,
    maxDepth,
    root,
    targetName,
    dtype,
  );
}
