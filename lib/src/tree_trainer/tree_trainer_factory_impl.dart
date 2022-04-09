import 'package:ml_algo/src/tree_trainer/decision_tree_trainer.dart';
import 'package:ml_algo/src/tree_trainer/leaf_detector/leaf_detector_factory.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_factory.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_type.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_factory.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_factory.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:quiver/iterables.dart';

class TreeTrainerFactoryImpl implements TreeTrainerFactory {
  const TreeTrainerFactoryImpl(
    this._leafDetectorFactory,
    this._leafLabelFactoryFactory,
    this._splitSelectorFactory,
  );

  final TreeLeafDetectorFactory _leafDetectorFactory;
  final TreeLeafLabelFactoryFactory _leafLabelFactoryFactory;
  final TreeSplitSelectorFactory _splitSelectorFactory;

  @override
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
  ) {
    final targetIdx = enumerate(samples.header)
        .firstWhere((indexedName) => indexedName.value == targetName)
        .index;
    final featuresIndexedSeries = enumerate(samples.series)
        .where((indexed) => indexed.index != targetIdx);
    final featureIdxToUniqueValues = Map.fromEntries(
      featuresIndexedSeries.where((indexed) => indexed.value.isDiscrete).map(
          (indexed) => MapEntry(
              indexed.index,
              indexed.value.discreteValues
                  .map((dynamic value) => value as num)
                  .toList(growable: false))),
    );
    final leafDetector = _leafDetectorFactory.create(
        leafAssessorType, minErrorOnNode, minSamplesCount, maxDepth);

    final leafLabelFactory =
        _leafLabelFactoryFactory.createByType(leafLabelFactoryType);

    final splitSelector = _splitSelectorFactory.createByType(
        splitSelectorType, splitAssessorType, splitterType);

    switch (type) {
      case TreeTrainerType.decision:
        return DecisionTreeTrainer(
          featuresIndexedSeries.map((indexed) => indexed.index),
          targetIdx,
          featureIdxToUniqueValues,
          leafDetector,
          leafLabelFactory,
          splitSelector,
        );

      default:
        throw UnsupportedError('Tree solver type $type is unsupported');
    }
  }
}
