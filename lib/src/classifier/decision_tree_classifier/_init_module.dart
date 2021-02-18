import 'package:get_it/get_it.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/_injector.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory_impl.dart';
import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator_factory.dart';
import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator_factory_impl.dart';
import 'package:ml_algo/src/di/common/init_common_module.dart';
import 'package:ml_algo/src/extensions/get_it.dart';
import 'package:ml_algo/src/tree_trainer/leaf_detector/leaf_detector_factory.dart';
import 'package:ml_algo/src/tree_trainer/leaf_detector/leaf_detector_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_factory.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor_factory.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_factory.dart';
import 'package:ml_algo/src/tree_trainer/split_selector/split_selector_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_factory.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_factory.dart';
import 'package:ml_algo/src/tree_trainer/tree_trainer_factory_impl.dart';

GetIt initDecisionTreeModule() {
  initCommonModule();

  return decisionTreeModule
    ..registerSingletonIf<DistributionCalculatorFactory>(
        const DistributionCalculatorFactoryImpl())

    ..registerSingletonIf<NominalTreeSplitterFactory>(
        const NominalTreeSplitterFactoryImpl())

    ..registerSingletonIf<NumericalTreeSplitterFactory>(
        const NumericalTreeSplitterFactoryImpl())

    ..registerSingletonIf<TreeSplitAssessorFactory>(
        const TreeSplitAssessorFactoryImpl())

    ..registerSingletonIf<TreeSplitterFactory>(
        TreeSplitterFactoryImpl(
          decisionTreeModule.get<TreeSplitAssessorFactory>(),
          decisionTreeModule.get<NominalTreeSplitterFactory>(),
          decisionTreeModule.get<NumericalTreeSplitterFactory>(),
        ))

    ..registerSingletonIf<TreeSplitSelectorFactory>(
        TreeSplitSelectorFactoryImpl(
          decisionTreeModule.get<TreeSplitAssessorFactory>(),
          decisionTreeModule.get<TreeSplitterFactory>(),
        ))

    ..registerSingletonIf<TreeLeafDetectorFactory>(
        TreeLeafDetectorFactoryImpl(
          decisionTreeModule.get<TreeSplitAssessorFactory>(),
        ))

    ..registerSingletonIf<TreeLeafLabelFactoryFactory>(
        TreeLeafLabelFactoryFactoryImpl(
          decisionTreeModule
              .get<DistributionCalculatorFactory>(),
        ))

    ..registerSingletonIf<TreeTrainerFactory>(
        TreeTrainerFactoryImpl(
          decisionTreeModule.get<TreeLeafDetectorFactory>(),
          decisionTreeModule.get<TreeLeafLabelFactoryFactory>(),
          decisionTreeModule.get<TreeSplitSelectorFactory>(),
        ))

    ..registerSingletonIf<DecisionTreeClassifierFactory>(
        DecisionTreeClassifierFactoryImpl(
          decisionTreeModule.get<TreeTrainerFactory>(),
        ));
}
