import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator_factory.dart';
import 'package:ml_algo/src/decision_tree_solver/leaf_label/leaf_label_factory.dart';
import 'package:ml_algo/src/decision_tree_solver/leaf_label/leaf_label_factory_factory.dart';
import 'package:ml_algo/src/decision_tree_solver/leaf_label/leaf_label_factory_type.dart';
import 'package:ml_algo/src/decision_tree_solver/leaf_label/majority_leaf_label_factory.dart';

class DecisionTreeLeafLabelFactoryFactoryImpl implements
    DecisionTreeLeafLabelFactoryFactory {

  DecisionTreeLeafLabelFactoryFactoryImpl(this._distributionCalculatorFactory);

  final SequenceElementsDistributionCalculatorFactory
    _distributionCalculatorFactory;

  @override
  DecisionTreeLeafLabelFactory createByType(
      DecisionTreeLeafLabelFactoryType type) {

    final distributionCalculator = _distributionCalculatorFactory.create();

    switch(type) {
      case DecisionTreeLeafLabelFactoryType.majority:
        return MajorityDecisionTreeLeafLabelFactory(distributionCalculator);

      default:
        throw UnsupportedError('Decision tree leaf label factory type $type '
            'is not supported');
    }
  }
}
