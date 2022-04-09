import 'package:ml_algo/src/tree_trainer/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';

abstract class TreeLeafDetectorFactory {
  TreeLeafDetector create(
    TreeAssessorType assessorType,
    num minErrorOnNode,
    int minSamplesCount,
    int maxDepth,
  );
}
