import 'package:ml_algo/src/tree_solver/split_assessor/split_assessor_type.dart';
import 'package:ml_algo/src/tree_solver/split_selector/split_selector.dart';
import 'package:ml_algo/src/tree_solver/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/tree_solver/splitter/splitter_type.dart';

abstract class TreeSplitSelectorFactory {
  TreeSplitSelector createByType(
      TreeSplitSelectorType type,
      TreeSplitAssessorType assessorType,
      TreeSplitterType splitterType,
  );
}
