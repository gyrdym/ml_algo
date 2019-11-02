import 'package:ml_algo/src/decision_tree_solver/split_assessor/split_assessor_factory.dart';
import 'package:ml_algo/src/decision_tree_solver/split_assessor/split_assessor_type.dart';
import 'package:ml_algo/src/decision_tree_solver/splitter/greedy_splitter.dart';
import 'package:ml_algo/src/decision_tree_solver/splitter/nominal_splitter/nominal_splitter_factory.dart';
import 'package:ml_algo/src/decision_tree_solver/splitter/numerical_splitter/numerical_splitter_factory.dart';
import 'package:ml_algo/src/decision_tree_solver/splitter/splitter.dart';
import 'package:ml_algo/src/decision_tree_solver/splitter/splitter_factory.dart';
import 'package:ml_algo/src/decision_tree_solver/splitter/splitter_type.dart';

class DecisionTreeSplitterFactoryImpl implements DecisionTreeSplitterFactory {
  DecisionTreeSplitterFactoryImpl(
      this._assessorFactory,
      this._nominalSplitterFactory,
      this._numericalSplitterFactory,
  );

  final DecisionTreeSplitAssessorFactory _assessorFactory;
  final NominalDecisionTreeSplitterFactory _nominalSplitterFactory;
  final NumericalDecisionTreeSplitterFactory _numericalSplitterFactory;

  @override
  DecisionTreeSplitter createByType(DecisionTreeSplitterType type,
      DecisionTreeSplitAssessorType assessorType) {

    final assessor = _assessorFactory.createByType(assessorType);
    final numericalSplitter = _numericalSplitterFactory.create();
    final nominalSplitter = _nominalSplitterFactory.create();

    switch (type) {
      case DecisionTreeSplitterType.greedy:
        return GreedyDecisionTreeSplitter(assessor, numericalSplitter,
            nominalSplitter);

      default:
        throw UnsupportedError('Decision tree splitter type $type is not '
            'supported');
    }
  }
}