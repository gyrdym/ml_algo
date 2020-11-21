import 'package:inject/inject.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_factory_impl.dart';
import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator_factory.dart';
import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator_factory_impl.dart';
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

@module
class DecisionTreeClassifierModule {
  static DecisionTreeClassifierModule getInstance() =>
      _instance ??= DecisionTreeClassifierModule();
  static DecisionTreeClassifierModule _instance;

  @provide
  @singleton
  DistributionCalculatorFactory provideDistributionCalculatorFactory() =>
      const DistributionCalculatorFactoryImpl();

  @provide
  @singleton
  NominalTreeSplitterFactory provideNominalTreeSplitterFactory() =>
      const NominalTreeSplitterFactoryImpl();

  @provide
  @singleton
  NumericalTreeSplitterFactory provideNumericalTreeSplitterFactory() =>
      const NumericalTreeSplitterFactoryImpl();

  @provide
  @singleton
  TreeSplitAssessorFactory provideTreeSplitAssessorFactory() =>
      const TreeSplitAssessorFactoryImpl();

  @provide
  @singleton
  TreeSplitterFactory provideTreeSplitterFactory(
      TreeSplitAssessorFactory treeSplitAssessorFactory,
      NominalTreeSplitterFactory nominalTreeSplitterFactory,
      NumericalTreeSplitterFactory numericalTreeSplitterFactory,
  ) => TreeSplitterFactoryImpl(
    treeSplitAssessorFactory,
    nominalTreeSplitterFactory,
    numericalTreeSplitterFactory,
  );

  @provide
  @singleton
  TreeSplitSelectorFactory provideTreeSplitSelectorFactory(
      TreeSplitAssessorFactory treeSplitAssessorFactory,
      TreeSplitterFactory treeSplitterFactory
  ) => TreeSplitSelectorFactoryImpl(
    treeSplitAssessorFactory,
    treeSplitterFactory,
  );

  @provide
  @singleton
  TreeLeafDetectorFactory provideTreeLeafDetectorFactory(
      TreeSplitAssessorFactory treeSplitAssessorFactory,
  ) => TreeLeafDetectorFactoryImpl(treeSplitAssessorFactory);

  @provide
  @singleton
  TreeLeafLabelFactoryFactory provideTreeLeafLabelFactoryFactory(
      DistributionCalculatorFactory distributionCalculatorFactory,
  ) => TreeLeafLabelFactoryFactoryImpl(distributionCalculatorFactory);

  @provide
  @singleton
  TreeTrainerFactory provideTreeTrainerFactory(
      TreeLeafDetectorFactory treeLeafDetectorFactory,
      TreeLeafLabelFactoryFactory treeLeafLabelFactoryFactory,
      TreeSplitSelectorFactory treeSplitSelectorFactory,
  ) => TreeTrainerFactoryImpl(
    treeLeafDetectorFactory,
    treeLeafLabelFactoryFactory,
    treeSplitSelectorFactory,
  );

  @provide
  @singleton
  DecisionTreeClassifierFactory provideDecisionTreeClassifierFactory(
      TreeTrainerFactory treeTrainerFactory,
  ) => DecisionTreeClassifierFactoryImpl(treeTrainerFactory);
}
