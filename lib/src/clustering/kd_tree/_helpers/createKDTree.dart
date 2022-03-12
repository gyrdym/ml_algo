import 'package:ml_algo/src/clustering/kd_tree/_init_module.dart';
import 'package:ml_algo/src/clustering/kd_tree/_injector.dart';
import 'package:ml_algo/src/clustering/kd_tree/kd_tree_impl.dart';
import 'package:ml_algo/src/clustering/kd_tree/kd_tree_trainer.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

KDTreeImpl createKDTree(
  DataFrame samples,
  int leafSize,
  DType dtype,
) {
  initKDTreeModule();

  final splitter = kdTreeInjector.get<NumericalTreeSplitter>();
  final trainer = KDTreeTrainer(leafSize, splitter);
  final root = trainer.train(samples.toMatrix(dtype));

  return KDTreeImpl(root, leafSize, dtype);
}
