import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_type.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_type.dart';

abstract class TreeTrainerFactory {
  TreeTrainer createByType(
      TreeTrainerType type,
      Iterable<int> featureIndices,
      int targetIdx,
      Map<int, List<num>> featureIdxToUniqueValues,
      num minErrorOnNode,
      int minSamplesCount,
      int maxDepth,
      TreeSplitAssessorType assessorType,
      TreeLeafLabelFactoryType leafLabelFactoryType,
      TreeSplitSelectorType splitSelectorType,
      TreeSplitAssessorType splitAssessorType,
      TreeSplitterType splitterType,
  );
}
