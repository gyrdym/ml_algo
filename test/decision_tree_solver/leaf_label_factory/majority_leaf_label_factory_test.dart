import 'dart:collection';

import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator.dart';
import 'package:ml_algo/src/decision_tree_solver/leaf_label_factory/majority_leaf_label_factory.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('MajorityLeafLabelFactory', () {
    test('should create decision tree leaf label - label should be a majority '
        'class label among the whole observation matrix', () {

      final observations = Matrix.fromList([
        [10, 20, 30, 0],
      ]);

      final firstClassLabel = 0.0;
      final firstClassProbability = 0.7;

      final secondClassLabel = 2.0;
      final secondClassProbability = 0.1;

      final thirdClassLabel = 3.0;
      final thirdClassProbability = 0.2;

      final distribution = HashMap<double, double>.from(<double, double>{
        firstClassLabel: firstClassProbability,
        secondClassLabel: secondClassProbability,
        thirdClassLabel: thirdClassProbability,
      });
      final distributionCalculator = createDistributionCalculator(
          observations.getColumn(3),
          distribution,
      );
      final labelFactory = MajorityDecisionTreeLeafLabelFactory(distributionCalculator);
      final label = labelFactory.create(observations, 3);

      expect(label.value, equals(0));
      expect(label.probability, equals(firstClassProbability));
    });

    test('should create decision tree leaf label if the observations contain '
        'only one class label', () {
      final observations = Matrix.fromList([
        [10, 20, 30, 3],
        [17, 10, 32, 3],
        [70, 80, 90, 3],
      ]);

      final classLabel = 3.0;
      final classProbability = 1.0;

      final distribution = HashMap<double, double>.from(<double, double>{
        classLabel: classProbability,
      });

      final distributionCalculator = createDistributionCalculator(
        observations.getColumn(3),
        distribution,
      );
      final labelFactory = MajorityDecisionTreeLeafLabelFactory(distributionCalculator);
      final label = labelFactory.create(observations, 3,);

      expect(label.value, equals(classLabel));
      expect(label.probability, equals(classProbability));
    });
  });
}

SequenceElementsDistributionCalculator createDistributionCalculator(
    Iterable<num> values, HashMap<double, double> distribution) {
  final distributionCalculator = DistributionCalculatorMock();

  when(distributionCalculator.calculate<double>(argThat(equals(values)), any))
      .thenReturn(distribution);

  return distributionCalculator;
}