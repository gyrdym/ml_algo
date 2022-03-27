import 'package:ml_algo/src/common/serializable/serializable.dart';
import 'package:ml_algo/src/retrieval/kd_tree/_helpers/create_kd_tree.dart';
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
  /// The bigger the number, the less
  /// effective search will be performed. If [leafSize] is equal to the number
  /// of [points], a regular KNN-search will take place.
  ///
  /// Extremely small [leafSize] leads to ineffective memory usage since in
  /// this case will be allocated a lot of kd-tree nodes
  int get leafSize;

  /// Data type for [points] matrix
  DType get dtype;

  Iterable<KDTreeNeighbour> query(Vector point, int k);
}
