import 'package:ml_algo/src/decision_tree_solver/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/decision_tree_solver/split_assessor/split_assessor_type.dart';

abstract class TreeLeafDetectorFactory {
  TreeLeafDetector create(
      TreeSplitAssessorType assessorType,
      double minErrorOnNode,
      int minSamplesCount,
      int maxDepth,
  );
}
