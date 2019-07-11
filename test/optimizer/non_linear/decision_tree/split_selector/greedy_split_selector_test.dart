import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_selector/greedy_split_selector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

import '../../../../test_utils/mocks.dart';

void main() {
  group('GreedySplitSelector', () {
    test('should find best split when splitting the samples by real number '
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

      final worstSplit = {
        DecisionTreeNodeMock(): Matrix.fromList([[1, 2, 3, 4, -1, -1, -1]]),
        DecisionTreeNodeMock(): Matrix.fromList([[10, 20, 30, 40, -1, -1, -1]]),
      };

      final worseSplit = {
        DecisionTreeNodeMock(): Matrix.fromList([[5, 6, 7, 8, -1, -1, 0]]),
        DecisionTreeNodeMock(): Matrix.fromList([[50, 60, 70, 80, -1, -1, 0]]),
      };

      final goodSplit = {
        DecisionTreeNodeMock(): Matrix.fromList([
          [15, 16, 17, 18, -1, 0, 0]
        ]),
        DecisionTreeNodeMock(): Matrix.fromList([
          [150, 160, 170, 180, -1, 0, 0]
        ]),
      };

      final bestSplit = {
        DecisionTreeNodeMock(): Matrix.fromList([
          [125, 126, 127, 128, 0, 0, 0],
        ]),
        DecisionTreeNodeMock(): Matrix.fromList([
          [1500, 1600, 1700, 1800, 0, 0, 0],
        ]),
      };

      final featuresRanges = [
        worstRange, bestRange, worseRange, goodRange,
      ];
      final assessor = SplitAssessorMock();
      final splitter = DecisionTreeSplitterMock();

      when(splitter.split(samples, worstRange, outcomesRange))
          .thenReturn(worstSplit);
      when(splitter.split(samples, worseRange, outcomesRange))
          .thenReturn(worseSplit);
      when(splitter.split(samples, goodRange, outcomesRange))
          .thenReturn(goodSplit);
      when(splitter.split(samples, bestRange, outcomesRange))
          .thenReturn(bestSplit);

      when(assessor.getAggregatedError(worstSplit.values,
          outcomesRange)).thenReturn(0.999);
      when(assessor.getAggregatedError(worseSplit.values,
          outcomesRange)).thenReturn(0.8);
      when(assessor.getAggregatedError(goodSplit.values,
          outcomesRange)).thenReturn(0.4);
      when(assessor.getAggregatedError(bestSplit.values,
          outcomesRange)).thenReturn(0.1);

      final selector = GreedySplitSelector(assessor, splitter);
      final actualSplit = selector
          .select(samples, outcomesRange, featuresRanges);

      expect(actualSplit.keys, equals(bestSplit.keys));
      expect(actualSplit.values, equals(bestSplit.values));
    });

    test('should find best split when splitting the samples by nominal '
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

      final badSplit = {
        DecisionTreeNodeMock(): Matrix.fromList([
          [1, 2, 3, 4, -1, -1, -1],
        ]),
        DecisionTreeNodeMock(): Matrix.fromList([
          [10, 20, 30, 40, -1, -1, -1],
        ]),
      };

      final goodSplit = {
        DecisionTreeNodeMock(): Matrix.fromList([
          [15, 16, 17, 18, -1, 0, 0],
        ]),
        DecisionTreeNodeMock(): Matrix.fromList([
          [150, 160, 170, 180, -1, 0, 0],
        ]),
      };

      final bestSplit = {
        DecisionTreeNodeMock(): Matrix.fromList([
          [1, 2, 3, 4, -1, -1, -1],
        ]),
        DecisionTreeNodeMock(): Matrix.fromList([
          [10, 20, 30, 40, -1, -1, -1],
        ]),
        DecisionTreeNodeMock(): Matrix.fromList([
          [12, 23, 33, 44, -1, -1, -1],
        ]),
      };

      final featuresRanges = {
        badFeatureRange, bestFeatureRange, goodFeatureRange,
      };

      final assessor = SplitAssessorMock();
      final splitter = DecisionTreeSplitterMock();
      final nominalValues = [
        Vector.fromList([1, 1, 1]),
        Vector.fromList([2, 2, 2]),
        Vector.fromList([3, 3, 3]),
      ];
      final rangeToNominalValues = {bestFeatureRange: nominalValues};

      when(splitter.split(samples, badFeatureRange, outcomesRange))
          .thenReturn(badSplit);
      when(splitter.split(samples, goodFeatureRange, outcomesRange))
          .thenReturn(goodSplit);
      when(splitter.split(samples, bestFeatureRange, outcomesRange,
          nominalValues)).thenReturn(bestSplit);

      when(assessor.getAggregatedError(badSplit.values,
          outcomesRange)).thenReturn(0.999);
      when(assessor.getAggregatedError(goodSplit.values,
          outcomesRange)).thenReturn(0.4);
      when(assessor.getAggregatedError(bestSplit.values,
          outcomesRange)).thenReturn(0.1);

      final selector = GreedySplitSelector(assessor, splitter);
      final actualSplit = selector.select(samples, outcomesRange, featuresRanges,
          rangeToNominalValues);

      expect(actualSplit.keys, equals(bestSplit.keys));
      expect(actualSplit.values, equals(bestSplit.values));
    });

    test('should select input matrix columns for splitting according to given '
        'feature columns ranges', () {
      final observations = Matrix.fromList([
        [10, 1, 1, 1, 50, 0, 0, 1],
        [12, 2, 2, 2, 52, 0, 1, 0],
        [11, 3, 3, 3, 53, 1, 0, 0],
      ]);

      final outcomesRange = ZRange.closed(4, 6);

      final goodFeatureRange = ZRange.singleton(0);
      final bestFeatureRange = ZRange.singleton(4);
      final ignoredFeatureRange = ZRange.closed(1, 3);

      final goodSplit = {
        DecisionTreeNodeMock(): Matrix.fromList([
          [1, 2, 3, 4, -1, -1, -1],
        ]),
        DecisionTreeNodeMock(): Matrix.fromList([
          [10, 20, 30, 40, -1, -1, -1],
        ]),
      };

      final bestSplit = {
        DecisionTreeNodeMock(): Matrix.fromList([
          [15, 16, 17, 18, -1, 0, 0],
        ]),
        DecisionTreeNodeMock(): Matrix.fromList([
          [150, 160, 170, 180, -1, 0, 0],
        ]),
      };

      final featuresColumnRanges = {
        goodFeatureRange, bestFeatureRange,
      };

      final assessor = SplitAssessorMock();
      final splitter = DecisionTreeSplitterMock();
      final categoricalValues = [
        Vector.fromList([1, 1, 1]),
        Vector.fromList([2, 2, 2]),
        Vector.fromList([3, 3, 3]),
      ];
      final rangeToCategoricalValues = {ignoredFeatureRange: categoricalValues};

      when(splitter.split(observations, goodFeatureRange, outcomesRange))
          .thenReturn(goodSplit);
      when(splitter.split(observations, bestFeatureRange, outcomesRange))
          .thenReturn(bestSplit);

      when(assessor.getAggregatedError(goodSplit.values,
          outcomesRange)).thenReturn(0.51);
      when(assessor.getAggregatedError(bestSplit.values,
          outcomesRange)).thenReturn(0.1);

      final selector = GreedySplitSelector(assessor, splitter);
      final stump = selector.select(observations, outcomesRange,
          featuresColumnRanges, rangeToCategoricalValues);

      expect(stump.keys, equals(bestSplit.keys));
      expect(stump.values, equals(bestSplit.values));

      verifyNever(splitter.split(observations, ignoredFeatureRange,
          outcomesRange));
    });
  });
}
