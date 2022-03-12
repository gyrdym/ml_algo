import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_impl.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/intermediate_tree_node_factory.dart';

class NumericalTreeSplitterFactoryImpl implements NumericalTreeSplitterFactory {
  const NumericalTreeSplitterFactoryImpl(this._treeNodeFactory);

  final IntermediateTreeNodeFactory _treeNodeFactory;

  @override
  NumericalTreeSplitter create() => NumericalTreeSplitterImpl(_treeNodeFactory);
}
