import 'package:ml_algo/src/tree_trainer/split_selector/split_selector.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';

abstract class TreeSplitSelectorFactory {
  TreeSplitSelector createByType(
    TreeSplitSelectorType type,
    TreeAssessorType assessorType,
    TreeSplitterType splitterType,
  );
}
