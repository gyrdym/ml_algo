import 'package:ml_algo/src/decision_tree_solver/splitter/numerical_splitter/numerical_splitter.dart';
import 'package:ml_algo/src/decision_tree_solver/splitter/numerical_splitter/numerical_splitter_factory.dart';
import 'package:ml_algo/src/decision_tree_solver/splitter/numerical_splitter/numerical_splitter_impl.dart';

class NumericalDecisionTreeSplitterFactoryImpl implements
    NumericalDecisionTreeSplitterFactory {

  const NumericalDecisionTreeSplitterFactoryImpl();

  @override
  NumericalDecisionTreeSplitter create() =>
      const NumericalDecisionTreeSplitterImpl();
}
