import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_node.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class _Split {
  _Split(this.left, this.right, this.midPoint);

  final Matrix left;
  final Matrix right;
  final Vector midPoint;
}

class KDTreeBuilder {
  KDTreeBuilder(this._leafSize);

  final int _leafSize;

  KDTreeNode train(Matrix points) {
    final isLeaf = points.rowsNum <= _leafSize;

    if (isLeaf) {
      return KDTreeNode(points: points);
    }

    final splitIdx = _getSplitIdx(points);
    final splitValue = points.getColumn(splitIdx).median();
    final split = _splitPoints(points, splitIdx, splitValue);

    return KDTreeNode(
      points: Matrix.fromRows([split.midPoint], dtype: split.midPoint.dtype),
      splitIndex: splitIdx,
      left: train(split.left),
      right: train(split.right),
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

  _Split _splitPoints(Matrix points, int splitIdx, num splitValue) {
    final left = <Vector>[];
    final right = <Vector>[];
    Vector? midPoint;

    for (var i = 0; i < points.rowsNum; i++) {
      final point = points[i];

      if (point[splitIdx] < splitValue) {
        left.add(point);
        continue;
      }

      if (midPoint == null || point[splitIdx] < midPoint[splitIdx]) {
        midPoint = point;
      }

      right.add(point);
    }

    return _Split(Matrix.fromRows(left, dtype: points.dtype),
        Matrix.fromRows(right, dtype: points.dtype), midPoint!);
  }
}
