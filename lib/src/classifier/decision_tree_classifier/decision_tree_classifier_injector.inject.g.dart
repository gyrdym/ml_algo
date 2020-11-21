import 'decision_tree_classifier_injector.dart' as _i1;
import 'decision_tree_classifier_module.dart' as _i2;
import '../../tree_trainer/split_assessor/split_assessor_factory.dart' as _i3;
import '../../tree_trainer/leaf_detector/leaf_detector_factory.dart' as _i4;
import '../../common/distribution_calculator/distribution_calculator_factory.dart'
    as _i5;
import '../../tree_trainer/leaf_label/leaf_label_factory_factory.dart' as _i6;
import '../../tree_trainer/splitter/nominal_splitter/nominal_splitter_factory.dart'
    as _i7;
import '../../tree_trainer/splitter/numerical_splitter/numerical_splitter_factory.dart'
    as _i8;
import '../../tree_trainer/splitter/splitter_factory.dart' as _i9;
import '../../tree_trainer/split_selector/split_selector_factory.dart' as _i10;
import '../../tree_trainer/tree_trainer_factory.dart' as _i11;
import 'decision_tree_classifier_factory.dart' as _i12;
import 'dart:async' as _i13;

class DecisionTreeClassifierInjector$Injector
    implements _i1.DecisionTreeClassifierInjector {
  DecisionTreeClassifierInjector$Injector.fromModule(
      this._decisionTreeClassifierModule);

  final _i2.DecisionTreeClassifierModule _decisionTreeClassifierModule;

  _i3.TreeSplitAssessorFactory _singletonTreeSplitAssessorFactory;

  _i4.TreeLeafDetectorFactory _singletonTreeLeafDetectorFactory;

  _i5.DistributionCalculatorFactory _singletonDistributionCalculatorFactory;

  _i6.TreeLeafLabelFactoryFactory _singletonTreeLeafLabelFactoryFactory;

  _i7.NominalTreeSplitterFactory _singletonNominalTreeSplitterFactory;

  _i8.NumericalTreeSplitterFactory _singletonNumericalTreeSplitterFactory;

  _i9.TreeSplitterFactory _singletonTreeSplitterFactory;

  _i10.TreeSplitSelectorFactory _singletonTreeSplitSelectorFactory;

  _i11.TreeTrainerFactory _singletonTreeTrainerFactory;

  _i12.DecisionTreeClassifierFactory _singletonDecisionTreeClassifierFactory;

  static _i13.Future<_i1.DecisionTreeClassifierInjector> create(
      _i2.DecisionTreeClassifierModule decisionTreeClassifierModule) async {
    final injector = DecisionTreeClassifierInjector$Injector.fromModule(
        decisionTreeClassifierModule);

    return injector;
  }

  _i12.DecisionTreeClassifierFactory _createDecisionTreeClassifierFactory() =>
      _singletonDecisionTreeClassifierFactory ??= _decisionTreeClassifierModule
          .provideDecisionTreeClassifierFactory(_createTreeTrainerFactory());
  _i11.TreeTrainerFactory _createTreeTrainerFactory() =>
      _singletonTreeTrainerFactory ??=
          _decisionTreeClassifierModule.provideTreeTrainerFactory(
              _createTreeLeafDetectorFactory(),
              _createTreeLeafLabelFactoryFactory(),
              _createTreeSplitSelectorFactory());
  _i4.TreeLeafDetectorFactory _createTreeLeafDetectorFactory() =>
      _singletonTreeLeafDetectorFactory ??= _decisionTreeClassifierModule
          .provideTreeLeafDetectorFactory(_createTreeSplitAssessorFactory());
  _i3.TreeSplitAssessorFactory _createTreeSplitAssessorFactory() =>
      _singletonTreeSplitAssessorFactory ??=
          _decisionTreeClassifierModule.provideTreeSplitAssessorFactory();
  _i6.TreeLeafLabelFactoryFactory _createTreeLeafLabelFactoryFactory() =>
      _singletonTreeLeafLabelFactoryFactory ??=
          _decisionTreeClassifierModule.provideTreeLeafLabelFactoryFactory(
              _createDistributionCalculatorFactory());
  _i5.DistributionCalculatorFactory _createDistributionCalculatorFactory() =>
      _singletonDistributionCalculatorFactory ??=
          _decisionTreeClassifierModule.provideDistributionCalculatorFactory();
  _i10.TreeSplitSelectorFactory _createTreeSplitSelectorFactory() =>
      _singletonTreeSplitSelectorFactory ??=
          _decisionTreeClassifierModule.provideTreeSplitSelectorFactory(
              _createTreeSplitAssessorFactory(), _createTreeSplitterFactory());
  _i9.TreeSplitterFactory _createTreeSplitterFactory() =>
      _singletonTreeSplitterFactory ??=
          _decisionTreeClassifierModule.provideTreeSplitterFactory(
              _createTreeSplitAssessorFactory(),
              _createNominalTreeSplitterFactory(),
              _createNumericalTreeSplitterFactory());
  _i7.NominalTreeSplitterFactory _createNominalTreeSplitterFactory() =>
      _singletonNominalTreeSplitterFactory ??=
          _decisionTreeClassifierModule.provideNominalTreeSplitterFactory();
  _i8.NumericalTreeSplitterFactory _createNumericalTreeSplitterFactory() =>
      _singletonNumericalTreeSplitterFactory ??=
          _decisionTreeClassifierModule.provideNumericalTreeSplitterFactory();
  @override
  _i12.DecisionTreeClassifierFactory getClassifierFactory() =>
      _createDecisionTreeClassifierFactory();
}
