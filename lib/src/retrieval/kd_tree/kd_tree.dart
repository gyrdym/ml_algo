import 'package:ml_algo/src/common/serializable/serializable.dart';
import 'package:ml_algo/src/retrieval/kd_tree/helpers/create_kd_tree.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_impl.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_neighbour.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

/// KD-tree - an algorithm that provides efficient data retrieval. It splits
/// the whole searching space into partitions in binary tree form which means
/// that data querying on average will take O(log(n)) time
abstract class KDTree implements Serializable {
  factory KDTree(DataFrame points,
          {int leafSie = 10, DType dtype = DType.float32}) =>
      createKDTree(points, leafSie, dtype);

  factory KDTree.fromJson(Map<String, dynamic> json) =>
      KDTreeImpl.fromJson(json);

  /// Points which were used to build the kd-tree
  Matrix get points;

  /// A number of points on a leaf node.
  ///
  /// The bigger the number, the less effective search is. If [leafSize] is
  /// equal to the number of [points], a regular KNN-search will take place.
  ///
  /// Extremely small [leafSize] leads to ineffective memory usage since in
  /// this case a lot of kd-tree nodes will be allocated
  int get leafSize;

  /// Data type for [points] matrix
  DType get dtype;

  /// Returns [k] nearest neighbours for [point]
  ///
  /// The neighbour is represented by an index and the distance between [point]
  /// and the neighbour itself. The index is a zero-based index of a point in
  /// the source [points] matrix. Example:
  ///
  /// ```dart
  /// final data = DataFrame([
  ///   [21, 34, 22, 11],
  ///   [11, 33, 44, 55],
  ///   ...,
  /// ], headerExists: false);
  /// final kdTree = KDTree(data);
  /// final neighbours = kdTree.query([1, 2, 3, 4], 2);
  ///
  /// print(neighbours.index); // let's say, it outputs `3` which means that the nearest neighbour is kdTree.points[3]
  /// ```
  Iterable<KDTreeNeighbour> query(Vector point, int k);
}
