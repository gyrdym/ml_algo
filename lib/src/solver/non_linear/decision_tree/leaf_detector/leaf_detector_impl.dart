import 'package:ml_algo/src/solver/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/split_assessor/split_assessor.dart';
import 'package:ml_linalg/matrix.dart';

class LeafDetectorImpl implements LeafDetector {
  LeafDetectorImpl(this._assessor, this._minErrorOnNode,
      this._minSamplesCount, this._maxDepth);

  final SplitAssessor _assessor;
  final int _minSamplesCount;
  final double _minErrorOnNode;
  final int _maxDepth;

  @override
  bool isLeaf(Matrix samples, int targetIdx,
      Iterable<num> featureColumnIdxs, int treeDepth) {
    if (_maxDepth <= treeDepth) {
      return true;
    }
    if (featureColumnIdxs.isEmpty) {
      return true;
    }
    if (samples.rowsNum <= _minSamplesCount) {
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
