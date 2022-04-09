import 'package:ml_algo/src/tree_trainer/split_selector/greedy_split_selector.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_factory.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_factory.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';

class TreeSplitSelectorFactoryImpl implements TreeSplitSelectorFactory {
  TreeSplitSelectorFactoryImpl(
    this._assessorFactory,
    this._splitterFactory,
  );

  final TreeAssessorFactory _assessorFactory;
  final TreeSplitterFactory _splitterFactory;

  @override
  TreeSplitSelector createByType(
    TreeSplitSelectorType type,
    TreeAssessorType assessorType,
    TreeSplitterType splitterType,
  ) {
    final assessor = _assessorFactory.createByType(assessorType);
    final splitter = _splitterFactory.createByType(splitterType, assessorType);

    switch (type) {
      case TreeSplitSelectorType.greedy:
        return GreedyTreeSplitSelector(assessor, splitter);

      default:
        throw UnsupportedError('Decision tree splitter type $type is '
            'not supported');
    }
  }
}
