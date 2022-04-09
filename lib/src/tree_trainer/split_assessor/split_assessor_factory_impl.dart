import 'package:ml_algo/src/common/distribution_calculator/distribution_calculator.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/gini_index_split_assessor.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/majority_split_assessor.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor_factory.dart';
import 'package:ml_algo/src/tree_trainer/assessor_type/assessor_type.dart';

class TreeSplitAssessorFactoryImpl implements TreeSplitAssessorFactory {
  const TreeSplitAssessorFactoryImpl(this._distributionCalculator);

  final DistributionCalculator _distributionCalculator;

  @override
  TreeSplitAssessor createByType(TreeAssessorType type) {
    switch (type) {
      case TreeAssessorType.majority:
        return const MajorityTreeSplitAssessor();

      case TreeAssessorType.gini:
        return GiniIndexTreeSplitAssessor(_distributionCalculator);

      default:
        throw UnsupportedError('Decision tree split assessor type $type is not '
            'supported');
    }
  }
}
