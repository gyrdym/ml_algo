import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_node.dart';
import 'package:ml_linalg/matrix.dart';

class _Split {
  _Split(this.left, this.right, this.midPoint);

  final List<int> left;
  final List<int> right;
  final int midPoint;
}

class KDTreeBuilder {
  KDTreeBuilder(this._leafSize, this._points);

  final int _leafSize;
  final Matrix _points;

  KDTreeNode train() => _train(_points.rowIndices.toList());

  KDTreeNode _train(List<int> pointIndices) {
    final isLeaf = pointIndices.length <= _leafSize;
    final points = _points.sample(rowIndices: pointIndices);
    final splitIdx = _getSplitIdx(points);

    if (isLeaf) {
      return KDTreeNode(splitIndex: splitIdx, pointIndices: pointIndices);
    }

    final split = _splitPoints(pointIndices, splitIdx);

    return KDTreeNode(
      pointIndices: [split.midPoint],
      splitIndex: splitIdx,
      left: _train(split.left),
      right: _train(split.right),
    );
  }

  int _getSplitIdx(Matrix points) {
    final variances = points.variance();

    var colIdx = 0;
    var maxIdx = colIdx;
    var max = variances[maxIdx];

    variances.forEach((variance) {
      if (variance > max) {
        max = variance;
        maxIdx = colIdx;
      }
      colIdx++;
    });

    return maxIdx;
  }

  _Split _splitPoints(List<int> pointIndices, int splitIdx) {
    pointIndices.sort((firstIdx, secondIdx) =>
        _points[firstIdx][splitIdx].compareTo(_points[secondIdx][splitIdx]));

    final midPointIdx = (pointIndices.length / 2).floor();
    final midPoint = pointIndices[midPointIdx];
    final left = pointIndices.sublist(0, midPointIdx);
    final right = pointIndices.sublist(midPointIdx + 1);

    return _Split(left, right, midPoint);
  }
}
