import 'package:ml_algo/src/tree_solver/split_assessor/majority_split_assessor.dart';
import 'package:ml_algo/src/tree_solver/split_assessor/split_assessor_factory_impl.dart';
import 'package:ml_algo/src/tree_solver/split_assessor/split_assessor_type.dart';
import 'package:test/test.dart';

void main() {
  group('TreeSplitAssessorFactoryImpl', () {
    final factory = const TreeSplitAssessorFactoryImpl();

    test('should create MajorityTreeSplitAssessor', () {
      final type = TreeSplitAssessorType.majority;
      final assessor = factory.createByType(type);

      expect(assessor, isA<MajorityTreeSplitAssessor>());
    });

    test('should throw an error if null passed as a type', () {
      expect(() => factory.createByType(null), throwsUnsupportedError);
    });
  });
}
