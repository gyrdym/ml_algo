import 'package:ml_algo/src/decision_tree_solver/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/decision_tree_solver/leaf_detector/leaf_detector_factory.dart';
import 'package:ml_algo/src/decision_tree_solver/leaf_detector/leaf_detector_impl.dart';
import 'package:ml_algo/src/decision_tree_solver/split_assessor/split_assessor_factory.dart';
import 'package:ml_algo/src/decision_tree_solver/split_assessor/split_assessor_type.dart';

class DecisionTreeLeafDetectorFactoryImpl implements
    DecisionTreeLeafDetectorFactory {

  DecisionTreeLeafDetectorFactoryImpl(this._splitAssessorFactory);

  final DecisionTreeSplitAssessorFactory _splitAssessorFactory;

  @override
  DecisionTreeLeafDetector create(
      DecisionTreeSplitAssessorType assessorType,
      double minErrorOnNode,
      int minSamplesCount,
      int maxDepth,
  ) {
    final assessor = _splitAssessorFactory.createByType(assessorType);

    return DecisionTreeLeafDetectorImpl(assessor, minErrorOnNode,
        minSamplesCount, maxDepth);
  }
}