import 'package:ml_algo/src/di/dependencies.dart';
import 'package:ml_algo/src/tree_solver/leaf_label/leaf_label_factory_type.dart';
import 'package:ml_algo/src/tree_solver/split_assessor/split_assessor_type.dart';
import 'package:ml_algo/src/tree_solver/split_selector/split_selector_type.dart';
import 'package:ml_algo/src/tree_solver/splitter/splitter_type.dart';
import 'package:ml_algo/src/tree_solver/tree_solver.dart';
import 'package:ml_algo/src/tree_solver/tree_solver_factory.dart';
import 'package:ml_algo/src/tree_solver/tree_solver_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:quiver/iterables.dart';

TreeSolver createDecisionTreeSolver(
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

  final colIdxToUniqueValues = Map.fromEntries(
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

  final solverFactory = dependencies.getDependency<TreeSolverFactory>();

  return solverFactory.createByType(
    TreeSolverType.decision,
    samples.toMatrix(),
    featuresIndexedSeries.map((indexed) => indexed.index),
    targetIdx,
    colIdxToUniqueValues,
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
