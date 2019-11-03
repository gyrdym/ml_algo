import 'package:ml_algo/src/tree_solver/split_assessor/majority_split_assessor.dart';
import 'package:ml_algo/src/tree_solver/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/tree_solver/split_assessor/split_assessor_factory.dart';
import 'package:ml_algo/src/tree_solver/split_assessor/split_assessor_type.dart';

class TreeSplitAssessorFactoryImpl implements TreeSplitAssessorFactory {

  const TreeSplitAssessorFactoryImpl();

  @override
  TreeSplitAssessor createByType(TreeSplitAssessorType type) {
    switch (type) {
      case TreeSplitAssessorType.majority:
        return const MajorityTreeSplitAssessor();

      default:
        throw UnsupportedError('Decision tree split assessor type $type is not '
            'supported');
    }
  }
}
