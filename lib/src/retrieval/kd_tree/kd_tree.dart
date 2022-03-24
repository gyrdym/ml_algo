import 'package:ml_algo/src/common/serializable/serializable.dart';
import 'package:ml_algo/src/retrieval/kd_tree/_helpers/create_kd_tree.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_impl.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_node.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';

abstract class KDTree implements Serializable {
  factory KDTree(DataFrame samples,
          {int leafSie = 10, DType dtype = DType.float32}) =>
      createKDTree(samples, leafSie, dtype);

  factory KDTree.fromJson(Map<String, dynamic> json) => KDTreeImpl.fromJson(json);

  int get leafSize;
  KDTreeNode get root;
  DType get dtype;

  Iterable<Vector> query(Vector sample, int k);
}
