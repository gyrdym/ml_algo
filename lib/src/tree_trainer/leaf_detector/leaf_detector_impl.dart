import 'package:ml_algo/src/tree_trainer/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor.dart';
import 'package:ml_linalg/matrix.dart';

class TreeLeafDetectorImpl implements TreeLeafDetector {
  TreeLeafDetectorImpl(this._assessor, this._minErrorOnNode,
      this._minSamplesCount, this._maxDepth);

  final TreeAssessor _assessor;
  final int _minSamplesCount;
  final num _minErrorOnNode;
  final int _maxDepth;

  @override
  bool isLeaf(Matrix samples, int targetIdx, Iterable<num> featureColumnIdxs,
      int treeDepth) {
    if (_maxDepth <= treeDepth) {
      return true;
    }

    if (featureColumnIdxs.isEmpty) {
      return true;
    }

    if (samples.rowCount <= _minSamplesCount) {
      return true;
    }

    final outcomes = samples.getColumn(targetIdx);

    if (outcomes.unique().length == 1) {
      return true;
    }

    final errorOnNode = _assessor.getError(samples, targetIdx);

    return errorOnNode <= _minErrorOnNode;
  }
}
