import 'package:injector/injector.dart';
import 'package:ml_algo/src/clustering/kd_tree/_injector.dart';
import 'package:ml_algo/src/extensions/injector.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_impl.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/intermediate_tree_node_factory.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/kd_tree_node/kd_intermediate_tree_node_factory.dart';

Injector initKDTreeModule() {
  return kdTreeInjector
    ..registerSingletonIf<IntermediateTreeNodeFactory>(
        () => const KDIntermediateTreeNodeFactory())
    ..registerSingletonIf<NumericalTreeSplitter>(() =>
        NumericalTreeSplitterImpl(
            kdTreeInjector.get<IntermediateTreeNodeFactory>()));
}
