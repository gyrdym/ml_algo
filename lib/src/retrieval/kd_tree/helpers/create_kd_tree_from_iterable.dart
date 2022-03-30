import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_builder.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_constants.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_impl.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

KDTreeImpl createKDTreeFromIterable(
    Iterable<Iterable<num>> pointsSrc, int leafSize, DType dtype) {
  final points = Matrix.fromList(
      pointsSrc
          .map((row) => row.map((element) => element.toDouble()).toList())
          .toList(),
      dtype: dtype);
  final builder = KDTreeBuilder(leafSize, points);
  final root = builder.train();

  return KDTreeImpl(points, leafSize, root, dtype, kdTreeJsonSchemaVersion);
}
