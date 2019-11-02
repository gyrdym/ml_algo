import 'package:ml_algo/src/decision_tree_solver/split_assessor/split_assessor_factory.dart';
import 'package:ml_algo/src/decision_tree_solver/split_assessor/split_assessor_type.dart';
import 'package:ml_algo/src/decision_tree_solver/split_selector/greedy_split_selector.dart';
import 'package:ml_algo/src/decision_tree_solver/split_selector/split_selector.dart';
import 'package:ml_algo/src/decision_tree_solver/split_selector/split_selector_factory.dart';
import 'package:ml_algo/src/decision_tree_solver/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/decision_tree_solver/splitter/splitter_factory.dart';
import 'package:ml_algo/src/decision_tree_solver/splitter/splitter_type.dart';

class DecisionTreeSplitSelectorFactoryImpl implements DecisionTreeSplitSelectorFactory {
  DecisionTreeSplitSelectorFactoryImpl(
    this._assessorFactory,
    this._splitterFactory,
  );

  final DecisionTreeSplitAssessorFactory _assessorFactory;
  final DecisionTreeSplitterFactory _splitterFactory;

  @override
  DecisionTreeSplitSelector createByType(
      DecisionTreeSplitSelectorType type,
      DecisionTreeSplitAssessorType assessorType,
      DecisionTreeSplitterType splitterType,
  ) {
    final assessor = _assessorFactory.createByType(assessorType);
    final splitter = _splitterFactory.createByType(splitterType, assessorType);

    switch(type) {
      case DecisionTreeSplitSelectorType.greedy:
        return GreedyDecisionTreeSplitSelector(assessor, splitter);

      default:
        throw UnsupportedError('Decision tree splitter type $type is '
            'not supported');
    }
  }
}
