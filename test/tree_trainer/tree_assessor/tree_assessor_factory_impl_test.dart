import 'package:ml_algo/src/tree_trainer/tree_assessor/gini_index_tree_assessor.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/majority_tree_assessor.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';
import 'package:test/test.dart';

import '../../mocks.mocks.dart';

void main() {
  group('TreeAssessorFactoryImpl', () {
    final distributionCalculatorMock = MockDistributionCalculator();
    final factory = TreeAssessorFactoryImpl(distributionCalculatorMock);

    test('should create MajorityTreeSplitAssessor', () {
      final type = TreeAssessorType.majority;
      final assessor = factory.createByType(type);

      expect(assessor, isA<MajorityTreeAssessor>());
    });

    test('should create GiniIndexTreeSplitAssessor', () {
      final type = TreeAssessorType.gini;
      final assessor = factory.createByType(type);

      expect(assessor, isA<GiniIndexTreeAssessor>());
    });
  });
}
