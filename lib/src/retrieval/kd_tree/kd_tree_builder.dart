import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_node.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class _Split {
  _Split(this.left, this.right, this.midVector);

  final Matrix left;
  final Matrix right;
  final Vector midVector;
}

class KDTreeBuilder {
  KDTreeBuilder(this._leafSize);

  final int _leafSize;

  KDTreeNode train(Matrix samples) {
    final isLeaf = samples.rowsNum <= _leafSize;

    if (isLeaf) {
      return KDTreeNode(samples: samples);
    }

    final splittingIdx = _getSplittingIdx(samples);
    final splittingValue = samples.getColumn(splittingIdx).mean();
    final split = _splitSamples(samples, splittingIdx, splittingValue);

    return KDTreeNode(
      value: split.midVector,
      index: splittingIdx,
      left: train(split.left),
      right: train(split.right),
    );
  }

  int _getSplittingIdx(Matrix samples) {
    final variances = samples.variance();

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

  _Split _splitSamples(Matrix samples, int splittingIdx, num splittingValue) {
    final left = <Vector>[];
    final right = <Vector>[];
    Vector? midSample;

    for (var i = 0; i < samples.rowsNum; i++) {
      final row = samples[i];

      if (row[splittingIdx] < splittingValue) {
        left.add(row);
        continue;
      }

      if (midSample == null || row[splittingIdx] < midSample[splittingIdx]) {
        midSample = row;
      }

      right.add(row);
    }

    return _Split(Matrix.fromRows(left, dtype: samples.dtype),
        Matrix.fromRows(right, dtype: samples.dtype), midSample!);
  }
}
