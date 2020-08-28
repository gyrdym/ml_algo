import 'package:ml_algo/src/di/dependencies.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_type.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_factory.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:quiver/iterables.dart';

TreeTrainer createDecisionTreeTrainer(
    DataFrame samples,
    String targetName,
    num minErrorOnNode,
    int minSamplesCountOnNode,
    int maxDepth,
) {
  final targetIdx = enumerate(samples.header)
      .firstWhere((indexedName) => indexedName.value == targetName)
      .index;

  final featuresIndexedSeries = enumerate(samples.series)
      .where((indexed) => indexed.index != targetIdx);

  final featureIdxToUniqueValues = Map.fromEntries(
      featuresIndexedSeries
        .where((indexed) => indexed.value.isDiscrete)
        .map((indexed) => MapEntry(indexed.index, indexed
          .value
          .discreteValues
          .map((dynamic value) => value as num)
          .toList(growable: false),
        ),
      ),
  );

  final trainerFactory = dependencies.get<TreeTrainerFactory>();

  return trainerFactory.createByType(
    TreeTrainerType.decision,
    featuresIndexedSeries.map((indexed) => indexed.index),
    targetIdx,
    featureIdxToUniqueValues,
    minErrorOnNode,
    minSamplesCountOnNode,
    maxDepth,
    TreeSplitAssessorType.majority,
    TreeLeafLabelFactoryType.majority,
    TreeSplitSelectorType.greedy,
    TreeSplitAssessorType.majority,
    TreeSplitterType.greedy,
  );
}
