import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/gini_index_tree_assessor.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/majority_tree_assessor.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_factory.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';

class TreeAssessorFactoryImpl implements TreeAssessorFactory {
  const TreeAssessorFactoryImpl(this._distributionCalculator);

  final DistributionCalculator _distributionCalculator;

  @override
  TreeAssessor createByType(TreeAssessorType type) {
    switch (type) {
      case TreeAssessorType.majority:
        return const MajorityTreeAssessor();

      case TreeAssessorType.gini:
        return GiniIndexTreeAssessor(_distributionCalculator);

      default:
        throw UnsupportedError('Decision tree split assessor type $type is not '
            'supported');
    }
  }
}
