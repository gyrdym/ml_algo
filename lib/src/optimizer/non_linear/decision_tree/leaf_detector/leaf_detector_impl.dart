import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:xrange/zrange.dart';

class LeafDetectorImpl implements LeafDetector {
  LeafDetectorImpl(this._assessor, this._minErrorOnNode,
      this._minSamplesCount);

  final SplitAssessor _assessor;
  final int _minSamplesCount;
  final double _minErrorOnNode;

  @override
  bool isLeaf(Matrix samples, ZRange outcomesRange,
      Iterable<ZRange> featureColumnRanges) {
    if (featureColumnRanges.isEmpty) {
      return true;
    }
    if (samples.rowsNum <= _minSamplesCount) {
      return true;
    }
    final outcomes = samples.submatrix(columns: outcomesRange);
    if (outcomes.uniqueRows().rowsNum == 1) {
      return true;
    }
    final errorOnNode = _assessor.getError(samples, outcomesRange);
    return errorOnNode <= _minErrorOnNode;
  }
}
