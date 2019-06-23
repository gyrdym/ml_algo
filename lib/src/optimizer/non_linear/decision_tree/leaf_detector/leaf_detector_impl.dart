import 'package:ml_algo/src/optimizer/non_linear/decision_tree/assessor/stump_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:xrange/zrange.dart';

class LeafDetectorImpl implements LeafDetector {
  const LeafDetectorImpl(this._assessor, this._minErrorOnNode,
      this._maxNodesCount);

  final StumpAssessor _assessor;
  final int _maxNodesCount;
  final int _minErrorOnNode;

  @override
  bool isLeaf(Matrix observations, ZRange outcomesRange, int nodesCount) {
    if (nodesCount >= _maxNodesCount) {
      return true;
    }
    final outcomes = observations.submatrix(columns: outcomesRange);
    if (outcomes.uniqueRows().rowsNum == 1) {
      return true;
    }
    final errorOnNode = _assessor.getErrorOnNode(observations, outcomesRange);
    return errorOnNode <= _minErrorOnNode;
  }
}
