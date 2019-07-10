import 'package:ml_algo/src/optimizer/non_linear/decision_tree/best_stump_finder/greedy_split_finder.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_stump.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

import '../../../../test_utils/mocks.dart';

void main() {
  group('GreedyStumpFinder', () {
    test('should find best stump when splitting the samples by real number '
        'value', () {
      final samples = Matrix.fromList([
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
      final assessor = SplitAssessorMock();
      final selector = StumpFactoryMock();

      when(selector.create(samples, worstRange, outcomesRange))
          .thenReturn(worstStump);
      when(selector.create(samples, worseRange, outcomesRange))
          .thenReturn(worseStump);
      when(selector.create(samples, goodRange, outcomesRange))
          .thenReturn(goodStump);
      when(selector.create(samples, bestRange, outcomesRange))
          .thenReturn(bestStump);

      when(assessor.getAggregatedError(worstStump.outputSamples,
          outcomesRange)).thenReturn(0.999);
      when(assessor.getAggregatedError(worseStump.outputSamples,
          outcomesRange)).thenReturn(0.8);
      when(assessor.getAggregatedError(goodStump.outputSamples,
          outcomesRange)).thenReturn(0.4);
      when(assessor.getAggregatedError(bestStump.outputSamples,
          outcomesRange)).thenReturn(0.1);

      final finder = GreedySplitFinder(assessor, selector);
      final stump = finder.find(samples, outcomesRange, featuresRanges);

      expect(stump, equals(bestStump));
    });

    test('should find best stump when splitting the samples by nominal '
        'feature', () {
      final samples = Matrix.fromList([
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

      final assessor = SplitAssessorMock();
      final selector = StumpFactoryMock();
      final nominalValues = [
        Vector.fromList([1, 1, 1]),
        Vector.fromList([2, 2, 2]),
        Vector.fromList([3, 3, 3]),
      ];
      final rangeToNominalValues = {bestFeatureRange: nominalValues};

      when(selector.create(samples, badFeatureRange, outcomesRange))
          .thenReturn(badStump);
      when(selector.create(samples, goodFeatureRange, outcomesRange))
          .thenReturn(goodStump);
      when(selector.create(samples, bestFeatureRange, outcomesRange,
          nominalValues)).thenReturn(bestStump);

      when(assessor.getAggregatedError(badStump.outputSamples,
          outcomesRange)).thenReturn(0.999);
      when(assessor.getAggregatedError(goodStump.outputSamples,
          outcomesRange)).thenReturn(0.4);
      when(assessor.getAggregatedError(bestStump.outputSamples,
          outcomesRange)).thenReturn(0.1);

      final finder = GreedySplitFinder(assessor, selector);
      final stump = finder.find(samples, outcomesRange, featuresRanges,
          rangeToNominalValues);

      expect(stump, equals(bestStump));
    });

    test('should select input matrix columns according to given feature '
        'columns ranges', () {
      final observations = Matrix.fromList([
        [10, 1, 1, 1, 50, 0, 0, 1],
        [12, 2, 2, 2, 52, 0, 1, 0],
        [11, 3, 3, 3, 53, 1, 0, 0],
      ]);

      final outcomesRange = ZRange.closed(4, 6);

      final goodFeatureRange = ZRange.singleton(0);
      final bestFeatureRange = ZRange.singleton(4);
      final ignoredFeatureRange = ZRange.closed(1, 3);

      final goodStump = DecisionTreeStump(null, null, null, [
        Matrix.fromList([[1, 2, 3, 4, -1, -1, -1]]),
        Matrix.fromList([[10, 20, 30, 40, -1, -1, -1]]),
      ]);

      final bestStump = DecisionTreeStump(null, null, null, [
        Matrix.fromList([[15, 16, 17, 18, -1, 0, 0]]),
        Matrix.fromList([[150, 160, 170, 180, -1, 0, 0]]),
      ]);

      final featuresColumnRanges = [
        goodFeatureRange, bestFeatureRange,
      ];

      final assessor = SplitAssessorMock();
      final selector = StumpFactoryMock();
      final categoricalValues = [
        Vector.fromList([1, 1, 1]),
        Vector.fromList([2, 2, 2]),
        Vector.fromList([3, 3, 3]),
      ];
      final rangeToCategoricalValues = {ignoredFeatureRange: categoricalValues};

      when(selector.create(observations, goodFeatureRange, outcomesRange))
          .thenReturn(goodStump);
      when(selector.create(observations, bestFeatureRange, outcomesRange))
          .thenReturn(bestStump);

      when(assessor.getAggregatedError(goodStump.outputSamples,
          outcomesRange)).thenReturn(0.51);
      when(assessor.getAggregatedError(bestStump.outputSamples,
          outcomesRange)).thenReturn(0.1);

      final finder = GreedySplitFinder(assessor, selector);
      final stump = finder.find(observations, outcomesRange,
          featuresColumnRanges, rangeToCategoricalValues);

      expect(stump, equals(bestStump));

      verifyNever(selector.create(observations, ignoredFeatureRange,
          outcomesRange));
    });
  });
}
