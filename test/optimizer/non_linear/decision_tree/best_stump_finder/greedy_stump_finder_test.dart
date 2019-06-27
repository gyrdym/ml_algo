import 'package:ml_algo/src/optimizer/non_linear/decision_tree/best_stump_finder/greedy_stump_finder.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_stump.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

import '../../../../test_utils/mocks.dart';

void main() {
  group('GreedyStumpFinder', () {
    test('should find best stump when splitting the observations by real '
        'number value', () {
      final observations = Matrix.fromList([
        [10, 20, 30, 40, 0, 0, 1],
        [12, 22, 32, 42, 0, 1, 0],
        [11, 21, 31, 41, 1, 0, 0],
      ]);

      final outcomesRange = ZRange.closed(4, 6);

      final worstRange = ZRange.singleton(0);
      final worseRange = ZRange.singleton(2);
      final goodRange = ZRange.singleton(3);
      final bestRange = ZRange.singleton(1);

      final worstStump = DecisionTreeStump(null, null, null, [
        Matrix.fromList([[1, 2, 3, 4, -1, -1, -1]]),
        Matrix.fromList([[10, 20, 30, 40, -1, -1, -1]]),
      ]);

      final worseStump = DecisionTreeStump(null, null, null, [
        Matrix.fromList([[5, 6, 7, 8, -1, -1, 0]]),
        Matrix.fromList([[50, 60, 70, 80, -1, -1, 0]]),
      ]);

      final goodStump = DecisionTreeStump(null, null, null, [
        Matrix.fromList([[15, 16, 17, 18, -1, 0, 0]]),
        Matrix.fromList([[150, 160, 170, 180, -1, 0, 0]]),
      ]);

      final bestStump = DecisionTreeStump(null, null, null, [
        Matrix.fromList([[125, 126, 127, 128, 0, 0, 0]]),
        Matrix.fromList([[1500, 1600, 1700, 1800, 0, 0, 0]]),
      ]);

      final featuresRanges = [
        worstRange, bestRange, worseRange, goodRange,
      ];
      final assessor = StumpAssessorMock();
      final selector = StumpSelectorMock();

      when(selector.select(observations, worstRange, outcomesRange))
          .thenReturn(worstStump);
      when(selector.select(observations, worseRange, outcomesRange))
          .thenReturn(worseStump);
      when(selector.select(observations, goodRange, outcomesRange))
          .thenReturn(goodStump);
      when(selector.select(observations, bestRange, outcomesRange))
          .thenReturn(bestStump);

      when(assessor.getErrorOnStump(worstStump.outputObservations, outcomesRange))
          .thenReturn(0.999);
      when(assessor.getErrorOnStump(worseStump.outputObservations, outcomesRange))
          .thenReturn(0.8);
      when(assessor.getErrorOnStump(goodStump.outputObservations, outcomesRange))
          .thenReturn(0.4);
      when(assessor.getErrorOnStump(bestStump.outputObservations, outcomesRange))
          .thenReturn(0.1);

      final finder = GreedyStumpFinder(assessor, selector);
      final stump = finder.find(observations, outcomesRange, featuresRanges);

      expect(stump, equals(bestStump));
    });

    test('should find best stump when splitting the observations by '
        'categorical feature', () {
      final observations = Matrix.fromList([
        [10, 1, 1, 1, 50, 0, 0, 1],
        [12, 2, 2, 2, 52, 0, 1, 0],
        [11, 3, 3, 3, 53, 1, 0, 0],
      ]);

      final outcomesRange = ZRange.closed(4, 6);

      final badFeatureRange = ZRange.singleton(0);
      final goodFeatureRange = ZRange.singleton(4);
      final bestFeatureRange = ZRange.closed(1, 3);

      final badStump = DecisionTreeStump(null, null, null, [
        Matrix.fromList([[1, 2, 3, 4, -1, -1, -1]]),
        Matrix.fromList([[10, 20, 30, 40, -1, -1, -1]]),
      ]);

      final goodStump = DecisionTreeStump(null, null, null, [
        Matrix.fromList([[15, 16, 17, 18, -1, 0, 0]]),
        Matrix.fromList([[150, 160, 170, 180, -1, 0, 0]]),
      ]);

      final bestStump = DecisionTreeStump(null, null, null, [
        Matrix.fromList([[1, 2, 3, 4, -1, -1, -1]]),
        Matrix.fromList([[10, 20, 30, 40, -1, -1, -1]]),
        Matrix.fromList([[12, 23, 33, 44, -1, -1, -1]]),
      ]);

      final featuresRanges = [
        badFeatureRange, bestFeatureRange, goodFeatureRange,
      ];

      final assessor = StumpAssessorMock();
      final selector = StumpSelectorMock();
      final categoricalValues = [
        Vector.fromList([1, 1, 1]),
        Vector.fromList([2, 2, 2]),
        Vector.fromList([3, 3, 3]),
      ];
      final rangeToCategoricalValues = {bestFeatureRange: categoricalValues};

      when(selector.select(observations, badFeatureRange, outcomesRange))
          .thenReturn(badStump);
      when(selector.select(observations, goodFeatureRange, outcomesRange))
          .thenReturn(goodStump);
      when(selector.select(observations, bestFeatureRange, outcomesRange,
          categoricalValues)).thenReturn(bestStump);

      when(assessor.getErrorOnStump(badStump.outputObservations, outcomesRange))
          .thenReturn(0.999);
      when(assessor.getErrorOnStump(goodStump.outputObservations, outcomesRange))
          .thenReturn(0.4);
      when(assessor.getErrorOnStump(bestStump.outputObservations, outcomesRange))
          .thenReturn(0.1);

      final finder = GreedyStumpFinder(assessor, selector);
      final stump = finder.find(observations, outcomesRange, featuresRanges,
          rangeToCategoricalValues);

      expect(stump, equals(bestStump));
    });
  });
}
