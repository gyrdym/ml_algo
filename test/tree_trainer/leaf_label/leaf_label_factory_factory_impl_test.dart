import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_factory_impl.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label_factory_type.dart';
import 'package:ml_algo/src/tree_trainer/leaf_label/majority_leaf_label_factory.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('TreeLeafLabelFactoryFactoryImpl', () {
    final distributionCalculator = DistributionCalculatorMock();

    final distributionCalculatorFactoryMock =
      createDistributionCalculatorFactoryMock(distributionCalculator);

    final factory = TreeLeafLabelFactoryFactoryImpl(
        distributionCalculatorFactoryMock);

    test('should create a MajorityTreeLeafLabelFactory instance', () {
      final leafLabelFactoryType = TreeLeafLabelFactoryType.majority;
      final leafLabelFactory = factory.createByType(leafLabelFactoryType);

      expect(leafLabelFactory, isA<MajorityTreeLeafLabelFactory>());
    });
  });
}
