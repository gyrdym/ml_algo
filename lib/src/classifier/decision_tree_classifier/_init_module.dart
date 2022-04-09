import 'package:injector/injector.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/_injector.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory_impl.dart';
import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator.dart';
import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator_factory.dart';
import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator_factory_impl.dart';
import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator_impl.dart';
import 'package:ml_algo/src/di/common/init_common_module.dart';
import 'package:ml_algo/src/extensions/injector.dart';
import 'package:ml_algo/src/tree_trainer/leaf_detector/leaf_detector_factory.dart';
import 'package:ml_algo/src/tree_trainer/leaf_detector/leaf_detector_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_factory.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_factory.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_factory.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_factory.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_factory_impl.dart';

Injector initDecisionTreeModule() {
  initCommonModule();

  return decisionTreeInjector
    ..registerSingletonIf<DistributionCalculatorFactory>(
        () => const DistributionCalculatorFactoryImpl())
    ..registerSingletonIf<DistributionCalculator>(
        () => const DistributionCalculatorImpl())
    ..registerSingletonIf<NominalTreeSplitterFactory>(
        () => const NominalTreeSplitterFactoryImpl())
    ..registerSingletonIf<NumericalTreeSplitterFactory>(
        () => const NumericalTreeSplitterFactoryImpl())
    ..registerSingletonIf<TreeAssessorFactory>(() => TreeAssessorFactoryImpl(
        decisionTreeInjector.get<DistributionCalculator>()))
    ..registerSingletonIf<TreeSplitterFactory>(() => TreeSplitterFactoryImpl(
          decisionTreeInjector.get<TreeAssessorFactory>(),
          decisionTreeInjector.get<NominalTreeSplitterFactory>(),
          decisionTreeInjector.get<NumericalTreeSplitterFactory>(),
        ))
    ..registerSingletonIf<TreeSplitSelectorFactory>(
        () => TreeSplitSelectorFactoryImpl(
              decisionTreeInjector.get<TreeAssessorFactory>(),
              decisionTreeInjector.get<TreeSplitterFactory>(),
            ))
    ..registerSingletonIf<TreeLeafDetectorFactory>(
        () => TreeLeafDetectorFactoryImpl(
              decisionTreeInjector.get<TreeAssessorFactory>(),
            ))
    ..registerSingletonIf<TreeLeafLabelFactoryFactory>(
        () => TreeLeafLabelFactoryFactoryImpl(
              decisionTreeInjector.get<DistributionCalculatorFactory>(),
            ))
    ..registerSingletonIf<TreeTrainerFactory>(() => TreeTrainerFactoryImpl(
          decisionTreeInjector.get<TreeLeafDetectorFactory>(),
          decisionTreeInjector.get<TreeLeafLabelFactoryFactory>(),
          decisionTreeInjector.get<TreeSplitSelectorFactory>(),
        ))
    ..registerSingletonIf<DecisionTreeClassifierFactory>(
        () => DecisionTreeClassifierFactoryImpl(
              decisionTreeInjector.get<TreeTrainerFactory>(),
            ));
}
