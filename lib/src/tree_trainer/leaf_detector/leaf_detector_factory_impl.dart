import 'package:ml_algo/src/tree_trainer/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/tree_trainer/leaf_detector/leaf_detector_factory.dart';
import 'package:ml_algo/src/tree_trainer/leaf_detector/leaf_detector_impl.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_factory.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';

class TreeLeafDetectorFactoryImpl implements TreeLeafDetectorFactory {
  TreeLeafDetectorFactoryImpl(this._splitAssessorFactory);

  final TreeAssessorFactory _splitAssessorFactory;

  @override
  TreeLeafDetector create(
    TreeAssessorType assessorType,
    num minErrorOnNode,
    int minSamplesCount,
    int maxDepth,
  ) {
    final assessor = _splitAssessorFactory.createByType(assessorType);

    return TreeLeafDetectorImpl(
        assessor, minErrorOnNode, minSamplesCount, maxDepth);
  }
}
