import 'package:ml_algo/src/tree_trainer/assessor_type/assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_type.dart';

abstract class TreeSplitSelectorFactory {
  TreeSplitSelector createByType(
    TreeSplitSelectorType type,
    TreeAssessorType assessorType,
    TreeSplitterType splitterType,
  );
}
