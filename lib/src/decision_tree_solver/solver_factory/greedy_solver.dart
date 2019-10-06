import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator_impl.dart';
import 'package:ml_algo/src/decision_tree_solver/decision_tree_solver.dart';
import 'package:ml_algo/src/decision_tree_solver/leaf_detector/leaf_detector_impl.dart';
import 'package:ml_algo/src/decision_tree_solver/leaf_label_factory/majority_leaf_label_factory.dart';
import 'package:ml_algo/src/decision_tree_solver/split_assessor/majority_split_assessor.dart';
import 'package:ml_algo/src/decision_tree_solver/split_selector/greedy_split_selector.dart';
import 'package:ml_algo/src/decision_tree_solver/splitter/greedy_splitter.dart';
import 'package:ml_algo/src/decision_tree_solver/splitter/nominal_splitter/nominal_splitter_impl.dart';
import 'package:ml_algo/src/decision_tree_solver/splitter/numerical_splitter/numerical_splitter_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:quiver/iterables.dart';

DecisionTreeSolver createGreedySolver(
    DataFrame samples,
    String targetName,
    double minErrorOnNode,
    int minSamplesCountOnNode,
    int maxDepth,
) {
  final assessor = const MajoritySplitAssessor();

  final numericalSplitter = const NumericalSplitterImpl();
  final nominalSplitter = const NominalSplitterImpl();
  final splitter = GreedySplitter(assessor, numericalSplitter, nominalSplitter);

  final distributionCalculator =
    const SequenceElementsDistributionCalculatorImpl();

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

  final leafDetector = LeafDetectorImpl(assessor, minErrorOnNode,
    minSamplesCountOnNode, maxDepth);
  final leafLabelFactory = MajorityLeafLabelFactory(distributionCalculator);
  final splitSelector = GreedySplitSelector(assessor, splitter);

  return DecisionTreeSolver(
    samples.toMatrix(),
    featuresIndexedSeries.map((indexed) => indexed.index),
    targetIdx,
    colIdxToUniqueValues,
    leafDetector,
    leafLabelFactory,
    splitSelector,
  );
}
