import 'package:ml_algo/src/classifier/asessable_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/decision_tree/decision_tree.dart';
import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator_impl.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/best_stump_finder/greedy_stump_finder.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector_impl.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/majority_leaf_label_factory.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_assessor/majority_split_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_factory/greedy_stump_factory.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_factory/observations_splitter/samples_splitter_impl.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

class GreedyDecisionTreeClassifier with AssessableClassifierMixin
    implements DecisionTreeClassifier {

  GreedyDecisionTreeClassifier(DataSet data, double minError,
      int minSamplesCount) :
        _optimizer = DecisionTreeOptimizer(
            data.toMatrix(),
            data.columnRanges,
            data.outcomeRange,
            data.rangeToEncoded,
            LeafDetectorImpl(
              const MajoritySplitAssessor(),
              minError,
              minSamplesCount,
            ),
            MajorityLeafLabelFactory(
              const SequenceElementsDistributionCalculatorImpl(),
            ),
            GreedyStumpFinder(
              const MajoritySplitAssessor(),
              GreedyStumpFactory(
                const MajoritySplitAssessor(),
                const SamplesSplitterImpl(),
              ),
            ),
        );

  final DecisionTreeOptimizer _optimizer;

  @override
  Matrix get classLabels => null;

  @override
  Matrix get coefficientsByClasses => null;

  @override
  Matrix predictClasses(Matrix features) {
    return null;
  }

  @override
  Matrix predictProbabilities(Matrix features) {
    return null;
  }

}
