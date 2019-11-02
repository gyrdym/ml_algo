import 'package:ml_algo/src/decision_tree_solver/split_assessor/majority_split_assessor.dart';
import 'package:ml_algo/src/decision_tree_solver/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/decision_tree_solver/split_assessor/split_assessor_factory.dart';
import 'package:ml_algo/src/decision_tree_solver/split_assessor/split_assessor_type.dart';

class DecisionTreeSplitAssessorFactoryImpl implements
    DecisionTreeSplitAssessorFactory {

  const DecisionTreeSplitAssessorFactoryImpl();

  @override
  DecisionTreeSplitAssessor createByType(DecisionTreeSplitAssessorType type) {
    switch (type) {
      case DecisionTreeSplitAssessorType.majority:
        return const MajorityDecisionTreeSplitAssessor();

      default:
        throw UnsupportedError('Decision tree split assessor type $type is not '
            'supported');
    }
  }
}
