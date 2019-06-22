import 'package:ml_algo/src/optimizer/non_linear/decision_tree/assessor/stump_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_linalg/matrix.dart';

class LeafDetectorImpl implements LeafDetector {
  const LeafDetectorImpl(this._assessor, this._minErrorOnNode,
      this._maxNodesCount);

  final StumpAssessor _assessor;
  final int _maxNodesCount;
  final int _minErrorOnNode;

  @override
  bool isLeaf(Matrix outcomes, int nodesCount) {
    if (nodesCount >= _maxNodesCount) {
      return true;
    }
    if (outcomes.uniqueRows().rowsNum == 1) {
      return true;
    }
    if (_assessor.getErrorOnNode(outcomes) <= _minErrorOnNode) {
      return true;
    }
    return false;
  }
}
