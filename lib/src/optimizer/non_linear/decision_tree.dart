import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class DecisionTreeNode {
  DecisionTreeNode(this.children);

  final Iterable<DecisionTreeNode> children;
}

class DecisionTreeOptimizer {
  DecisionTreeOptimizer(Matrix features, Matrix outcomes,
      [this._maxNodesCount, this._featuresRanges]) {
    _root = _createNode(features, outcomes, 0);
  }

  final int _maxNodesCount;
  final Iterable<ZRange> _featuresRanges;
  DecisionTreeNode _root;

  /// Builds a tree, where each node is a logical rule, that divides given data
  /// into several parts
  DecisionTreeNode _createNode(Matrix features, Matrix outcomes,
      int nodesCount) {
    final labels = outcomes.uniqueRows();
    if (_isLeaf(features, outcomes, labels, nodesCount)) {
      return DecisionTreeNode([]);
    }
    final range = _findSplittingFeatureValuesRange(features, outcomes);
    final data = features.submatrix(columns: range);
    final children = _getSplittingValues(data)
        .map((value) => _selectObservations(features, outcomes, range, value))
        .map((selected) => _createNode(selected.first, selected.last,
        nodesCount + 1));
    return DecisionTreeNode(children);
  }

  ZRange _findSplittingFeatureValuesRange(Matrix features, Matrix labels) {
    final errors = <double, List<ZRange>>{};
    for (final range in _featuresRanges) {
      final column = features.submatrix(columns: range);
      final stump = _getSplittingValues(column)
          .map((value) => _selectObservations(features, labels, range, value));
      final error = _getErrorOnStump(stump);
      errors.putIfAbsent(error, () => []);
      errors[error].add(range);
    }
    final sorted = errors.keys.toList(growable: false)
      ..sort();
    final minError = sorted.first;
    return errors[minError].first;
  }

  List<Vector> _getSplittingValues(Matrix column,
      [List<Vector> categoricalValues]) {
    if (categoricalValues?.isNotEmpty == true) {
      return categoricalValues;
    }

  }

  Iterable<Matrix> _selectObservations(Matrix features, Matrix outcomes,
      ZRange rangeToSplit, Vector value) {}

  bool _isLeaf(Matrix features, Matrix outcomes, Matrix labels,
      int nodesCount) {
    if (nodesCount >= _maxNodesCount) {
      return true;
    }
    if (labels.rowsNum == outcomes.rowsNum) {
      return true;
    }
    if (_isGoodQualityReached(outcomes, labels)) {
      return true;
    }
    return false;
  }

  bool _isGoodQualityReached(Matrix outcomes, Matrix labels) {

  }

  double _getErrorOnStump(Iterable<Iterable<Matrix>> data) {}
}
