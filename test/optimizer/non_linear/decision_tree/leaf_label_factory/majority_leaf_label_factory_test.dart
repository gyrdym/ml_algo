import 'dart:collection';

import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/majority_leaf_label_factory.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/observations_distribution_counter/distribution_counter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

import '../../../../test_utils/mocks.dart';

void main() {
  group('MajorityLeafLabelFactory', () {
    test('should create decision tree leaf label in case when the class '
        'labels are numbers - label should be a majority class label among the '
        'whole observation matrix', () {

      final observations = Matrix.fromList([
        [10, 20, 30, 0],
      ]);

      final outcomesColumnRange = ZRange.singleton(3);

      final firstClassLabel = 0.0;
      final firstClassLabelCount = 10000;

      final secondClassLabel = 2.0;
      final secondClassLabelCount = 50;

      final thirdClassLabel = 3.0;
      final thirdClassLabelCount = 1;

      final distribution = HashMap<double, int>.from(<double, int>{
        firstClassLabel: firstClassLabelCount,
        secondClassLabel: secondClassLabelCount,
        thirdClassLabel: thirdClassLabelCount,
      });
      final distributionCounter = createDistributionCounter<double>(
          observations.getColumn(3),
          distribution,
      );
      final labelFactory = MajorityLeafLabelFactory(distributionCounter);
      final label = labelFactory.create(observations, outcomesColumnRange,
          false);

      expect(label.numericalValue, equals(0));
    });

    test('should create decision tree leaf label in case when the class '
        'labels are numbers and the observations contain only one class '
        'label', () {
      final observations = Matrix.fromList([
        [10, 20, 30, 3],
      ]);
      final outcomesColumnRange = ZRange.singleton(3);

      final classLabel = 3.0;
      final classLabelCount = 20;

      final distribution = HashMap<double, int>.from(<double, int>{
        classLabel: classLabelCount,
      });

      final distributionCounter = createDistributionCounter<double>(
        observations.getColumn(3),
        distribution,
      );
      final labelFactory = MajorityLeafLabelFactory(distributionCounter);
      final label = labelFactory.create(observations, outcomesColumnRange,
          false);

      expect(label.numericalValue, equals(3));
    });

    test('should create decision tree leaf label in case when the class '
        'labels are vectors (encoded categories) - label should be a majority '
        'class label among the whole observation matrix', () {

      final observations = Matrix.fromList([
        [10, 20, 30, 0, 0, 1],
      ]);

      final outcomesColumnRange = ZRange.closed(3, 5);

      final firstClassLabel = Vector.fromList([0, 0, 1]);
      final firstClassLabelCount = 550;

      final secondClassLabel = Vector.fromList([1, 0, 0]);
      final secondClassLabelCount = 549;

      final thirdClassLabel = Vector.fromList([0, 1, 0]);
      final thirdClassLabelCount = 551;

      final distribution = HashMap<Vector, int>.from(<Vector, int>{
        firstClassLabel: firstClassLabelCount,
        secondClassLabel: secondClassLabelCount,
        thirdClassLabel: thirdClassLabelCount,
      });
      final distributionCounter = createDistributionCounter<Vector>(
        [Vector.fromList([0, 0, 1])],
        distribution,
      );
      final labelFactory = MajorityLeafLabelFactory(distributionCounter);
      final label = labelFactory.create(observations, outcomesColumnRange,
          true);

      expect(label.categoricalValue, equals(thirdClassLabel));
      expect(label.numericalValue, isNull);
    });

    test('should create decision tree leaf label in case when the class '
        'labels are vectors and the observations contain only one class '
        'label', () {
      final observations = Matrix.fromList([
        [10, 20, 30, 0, 1, 0],
      ]);
      final outcomesColumnRange = ZRange.closed(3, 5);

      final classLabel = Vector.fromList([0, 0, 1]);
      final classLabelCount = 121;

      final distribution = HashMap<Vector, int>.from(<Vector, int>{
        classLabel: classLabelCount,
      });

      final distributionCounter = createDistributionCounter<Vector>(
        [Vector.fromList([0, 1, 0])],
        distribution,
      );
      final labelFactory = MajorityLeafLabelFactory(distributionCounter);
      final label = labelFactory.create(observations, outcomesColumnRange,
          true);

      expect(label.categoricalValue, equals(classLabel));
      expect(label.numericalValue, isNull);
    });
  });
}

ObservationsDistributionCounter createDistributionCounter<T>(
    Iterable<T> values, HashMap<T, int> distribution) {
  final distributionCounter = ObservationsDistributionCounterMock();

  when(distributionCounter.count<T>(argThat(equals(values))))
      .thenReturn(distribution);

  return distributionCounter;
}