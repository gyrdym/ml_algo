import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_builder.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

KDTreeImpl createKDTree(DataFrame samples, int leafSize, DType dtype) {
  final builder = KDTreeBuilder(leafSize);
  final root = builder.train(samples.toMatrix(dtype));

  return KDTreeImpl(leafSize, root, dtype);
}
