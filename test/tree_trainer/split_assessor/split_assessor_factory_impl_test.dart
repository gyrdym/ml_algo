import 'package:ml_algo/src/tree_trainer/split_assessor/gini_index_split_assessor.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/majority_split_assessor.dart';
import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/assessor_type/assessor_type.dart';
import 'package:test/test.dart';

import '../../mocks.mocks.dart';

void main() {
  group('TreeSplitAssessorFactoryImpl', () {
    final distributionCalculatorMock = MockDistributionCalculator();
    final factory = TreeSplitAssessorFactoryImpl(distributionCalculatorMock);

    test('should create MajorityTreeSplitAssessor', () {
      final type = TreeAssessorType.majority;
      final assessor = factory.createByType(type);

      expect(assessor, isA<MajorityTreeSplitAssessor>());
    });

    test('should create GiniIndexTreeSplitAssessor', () {
      final type = TreeAssessorType.gini;
      final assessor = factory.createByType(type);

      expect(assessor, isA<GiniIndexTreeSplitAssessor>());
    });
  });
}
