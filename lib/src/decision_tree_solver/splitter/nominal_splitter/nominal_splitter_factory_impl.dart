import 'package:ml_algo/src/decision_tree_solver/splitter/nominal_splitter/nominal_splitter.dart';
import 'package:ml_algo/src/decision_tree_solver/splitter/nominal_splitter/nominal_splitter_factory.dart';
import 'package:ml_algo/src/decision_tree_solver/splitter/nominal_splitter/nominal_splitter_impl.dart';

class NominalDecisionTreeSplitterFactoryImpl implements
    NominalDecisionTreeSplitterFactory {

  const NominalDecisionTreeSplitterFactoryImpl();

  @override
  NominalDecisionTreeSplitter create() =>
      const NominalDecisionTreeSplitterImpl();
}