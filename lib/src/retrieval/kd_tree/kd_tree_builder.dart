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

    if (isLeaf) {
      return KDTreeNode(pointIndices: pointIndices);
    }

    final points = _points.sample(rowIndices: pointIndices);
    final splitIdx = _getSplitIdx(points);
    final splitValue = points.getColumn(splitIdx).median();
    final split = _splitPoints(pointIndices, splitIdx, splitValue);

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

  _Split _splitPoints(List<int> pointIndices, int splitIdx, num splitValue) {
    final left = <int>[];
    final right = <int>[];
    int? midPoint;

    for (var i = 0; i < pointIndices.length; i++) {
      final pointIndex = pointIndices[i];
      final point = _points[pointIndex];

      if (point[splitIdx] < splitValue) {
        left.add(pointIndex);
        continue;
      }

      if (midPoint == null || point[splitIdx] < _points[midPoint][splitIdx]) {
        midPoint = pointIndex;
      }

      right.add(pointIndex);
    }

    return _Split(left, right, midPoint!);
  }
}
