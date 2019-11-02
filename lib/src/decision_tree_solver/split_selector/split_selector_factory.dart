import 'package:ml_algo/src/decision_tree_solver/split_assessor/split_assessor_type.dart';
import 'package:ml_algo/src/decision_tree_solver/split_selector/split_selector.dart';
import 'package:ml_algo/src/decision_tree_solver/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/decision_tree_solver/splitter/splitter_type.dart';

abstract class DecisionTreeSplitSelectorFactory {
  DecisionTreeSplitSelector createByType(
      DecisionTreeSplitSelectorType type,
      DecisionTreeSplitAssessorType assessorType,
      DecisionTreeSplitterType splitterType,
  );
}
