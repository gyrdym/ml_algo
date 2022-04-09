import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_type.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

abstract class TreeTrainerFactory {
  TreeTrainer createByType(
    TreeTrainerType type,
    DataFrame samples,
    String targetName,
    num minErrorOnNode,
    int minSamplesCount,
    int maxDepth,
    TreeAssessorType leafAssessorType,
    TreeLeafLabelFactoryType leafLabelFactoryType,
    TreeSplitSelectorType splitSelectorType,
    TreeAssessorType splitAssessorType,
    TreeSplitterType splitterType,
  );
}
