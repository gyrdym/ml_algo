import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_linalg/matrix.dart';

abstract class TreeSplitSelector {
  /// Selects the best split of the [samples] and returns split results as a map
  /// where the key is a tree node and the value is a matrix to be processed
  /// further through the node
  ///
  /// Example:
  ///
  /// The [samples] are:
  ///
  /// [
  ///  [ 1,  2,  3,  4,  5,  6],
  ///  [11, 12, 13, 14, 15, 16],
  ///  [ 5,  6,  7,  8,  9, 10],
  /// ]
  ///
  /// Where [targetId] is `5`, which means that a column on the index 5 is a target one
  ///
  /// The selector established that the best split should be done through the
  /// 4th column:
  ///
  /// [
  ///  [4],
  ///  [14],
  ///  [8]
  /// ]
  ///
  /// according to the following predicate: every value less than 8
  /// goes to one node, and the rest values go to another node:
  ///
  /// {
  ///   <node_1>: [
  ///     [ 1,  2,  3,  4,  5,  6],
  ///   ],
  ///   <node_2>: [
  ///     [11, 12, 13, 14, 15, 16],
  //      [ 5,  6,  7,  8,  9, 10],
  ///   ],
  /// }
  Map<T, Matrix> select<T extends TreeNode>(
      Matrix samples, int targetId, Iterable<int> featuresColumnIdxs,
      [Map<int, List<num>> columnIdToUniqueValues]);
}
