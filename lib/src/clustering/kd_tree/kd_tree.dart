import 'package:ml_algo/src/clustering/kd_tree/_helpers/createKDTree.dart';
import 'package:ml_algo/src/clustering/kd_tree/kd_point.dart';
import 'package:ml_algo/src/common/serializable/serializable.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';

/// KD Tree (k-dimensional tree) - algorithm for dividing the feature space into
/// clusters in form of the binary tree. Once the tree is built, one can perform
/// efficient retrieval of the closest neighbours to the given vector
abstract class KDTree implements Serializable {
  factory KDTree(DataFrame samples,
          {int leafSize = 10, DType dtype = DType.float32}) =>
      createKDTree(samples, leafSize, dtype);

  /// A number of samples on the leaf node.
  ///
  /// Once a node gets a number of samples that is equal to [leafSize], the node
  /// becomes a leaf
  int get leafSize;

  /// A data type for all numerical values used in the algorithm
  DType get dtype;

  /// Returns [k] indices and distances of the closest vectors to [sample]
  Iterable<KDPoint> query(Vector sample, int k);
}
