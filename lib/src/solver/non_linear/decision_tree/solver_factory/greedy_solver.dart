import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator_impl.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/decision_tree_solver.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/leaf_detector/leaf_detector_impl.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/leaf_label_factory/majority_leaf_label_factory.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/split_assessor/majority_split_assessor.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/split_selector/greedy_split_selector.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/splitter/greedy_splitter.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/splitter/nominal_splitter/nominal_splitter_impl.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/splitter/numerical_splitter/numerical_splitter_impl.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:xrange/zrange.dart';

DecisionTreeSolver createGreedySolver(
    Matrix samples,
    Iterable<ZRange> columnRanges,
    ZRange outcomeRange,
    Map<ZRange, List<Vector>> rangeToEncoded,
    double minErrorOnNode,
    int minSamplesCountOnNode,
    int maxDepth) {

  final assessor = const MajoritySplitAssessor();

  final numericalSplitter = const NumericalSplitterImpl();
  final nominalSplitter = const NominalSplitterImpl();
  final splitter = GreedySplitter(
    assessor,
    numericalSplitter,
    nominalSplitter,
  );

  final distributionCalculator =
    const SequenceElementsDistributionCalculatorImpl();

  return DecisionTreeSolver(
    samples,
    columnRanges,
    outcomeRange,
    rangeToEncoded,
    LeafDetectorImpl(
      assessor,
      minErrorOnNode,
      minSamplesCountOnNode,
      maxDepth,
    ),
    MajorityLeafLabelFactory(
      distributionCalculator,
    ),
    GreedySplitSelector(
      assessor,
      splitter,
    ),
  );
}
