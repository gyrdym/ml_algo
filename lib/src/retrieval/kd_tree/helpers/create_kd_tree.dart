import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_builder.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_constants.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_impl.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_split_strategy.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

KDTreeImpl createKDTree(DataFrame pointsSrc, int leafSize, DType dtype,
    KDTreeSplitStrategy splitStrategy) {
  final points = pointsSrc.toMatrix(dtype);
  final builder = KDTreeBuilder(leafSize, points, splitStrategy);
  final root = builder.train();

  return KDTreeImpl(points, leafSize, root, dtype, kdTreeJsonSchemaVersion);
}
