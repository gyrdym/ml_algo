import 'package:ml_algo/src/optimizer/non_linear/decision_tree/assessor/stump_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:xrange/zrange.dart';

class LeafDetectorImpl implements LeafDetector {
  LeafDetectorImpl(this._assessor, this._minErrorOnNode, this._maxNodesCount,
      this._outcomesRange);

  final StumpAssessor _assessor;
  final int _maxNodesCount;
  final int _minErrorOnNode;
  final ZRange _outcomesRange;

  @override
  bool isLeaf(Matrix observations, int nodesCount) {
    final outcomes = observations.submatrix(columns: _outcomesRange);
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
