import 'package:injector/injector.dart';
import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator.dart';
import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator_impl.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/best_stump_finder/greedy_stump_finder.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector_impl.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/leaf_label_factory.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/majority_leaf_label_factory.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/nominal_splitter/nominal_splitter.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/nominal_splitter/nominal_splitter_impl.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/numerical_splitter/numerical_splitter.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/numerical_splitter/numerical_splitter_impl.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_assessor/majority_split_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_factory/greedy_stump_factory.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_factory/stump_factory.dart';

Injector getGreedyDecisionTreeDependencies(double minError,
    int minSamplesCount) => Injector()
    ..registerSingleton<SplitAssessor>((_) => const MajoritySplitAssessor())

    ..registerSingleton<SequenceElementsDistributionCalculator>(
            (_) => const SequenceElementsDistributionCalculatorImpl())

    ..registerSingleton<SplitAssessor>((_) => const MajoritySplitAssessor())

    ..registerSingleton<NumericalSplitter>(
            (_) => const NumericalSplitterImpl())

    ..registerSingleton<NominalSplitter>(
            (_) => const NominalSplitterImpl())

    ..registerSingleton<StumpFactory>((injector) =>
      GreedyStumpFactory(
        injector.getDependency<SplitAssessor>(),
        injector.getDependency<NumericalSplitter>(),
        injector.getDependency<NominalSplitter>(),
      ),
    )
    ..registerSingleton<LeafDetector>((injector) =>
        LeafDetectorImpl(
          injector.getDependency<SplitAssessor>(),
          minError,
          minSamplesCount,
      )
    )
    ..registerSingleton((injector) =>
      GreedyStumpFinder(
          injector.getDependency<SplitAssessor>(),
          injector.getDependency<StumpFactory>(),
      ),
    )
    ..registerSingleton<DecisionTreeLeafLabelFactory>((injector) {
      final distributionCalculator = injector
          .getDependency<SequenceElementsDistributionCalculator>();
      return MajorityLeafLabelFactory(distributionCalculator);
    });