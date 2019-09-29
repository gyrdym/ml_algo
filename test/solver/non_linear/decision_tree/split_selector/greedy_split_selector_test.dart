import 'package:ml_algo/src/decision_tree_solver/split_selector/greedy_split_selector.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../../../test_utils/mocks.dart';

void main() {
  group('GreedySplitSelector', () {
    test('should find the best split', () {
      final samples = Matrix.fromList([
        [10, 20, 30, 40, 1],
        [12, 22, 32, 42, 2],
        [11, 21, 31, 41, 3],
      ]);

      final targetColIdx = 4;

      final worstSplitIdx = 0;
      final worseSplitIdx = 2;
      final goodSplitIdx = 3;
      final bestSplitIdx = 1;

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
          [15, 16, 17, 18, -1, 0, 0],
        ]),
        DecisionTreeNodeMock(): Matrix.fromList([
          [150, 160, 170, 180, -1, 0, 0],
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

      final featuresIdxs = [
        worstSplitIdx, bestSplitIdx, worseSplitIdx, goodSplitIdx,
      ];
      final assessor = SplitAssessorMock();
      final splitter = DecisionTreeSplitterMock();

      when(splitter.split(samples, worstSplitIdx, targetColIdx))
          .thenReturn(worstSplit);
      when(splitter.split(samples, worseSplitIdx, targetColIdx))
          .thenReturn(worseSplit);
      when(splitter.split(samples, goodSplitIdx, targetColIdx))
          .thenReturn(goodSplit);
      when(splitter.split(samples, bestSplitIdx, targetColIdx))
          .thenReturn(bestSplit);

      when(assessor.getAggregatedError(worstSplit.values,
          targetColIdx)).thenReturn(0.999);
      when(assessor.getAggregatedError(worseSplit.values,
          targetColIdx)).thenReturn(0.8);
      when(assessor.getAggregatedError(goodSplit.values,
          targetColIdx)).thenReturn(0.4);
      when(assessor.getAggregatedError(bestSplit.values,
          targetColIdx)).thenReturn(0.1);

      final selector = GreedySplitSelector(assessor, splitter);
      final actualSplit = selector
          .select(samples, targetColIdx, featuresIdxs);

      expect(actualSplit.keys, equals(bestSplit.keys));
      expect(actualSplit.values, equals(bestSplit.values));
    });

    test('should select columns for splitting according to given feature '
        'columns indices', () {
      final observations = Matrix.fromList([
        [10, 1, 50, 1],
        [12, 2, 52, 2],
        [11, 3, 53, 3],
      ]);

      final targetColIdx = 3;

      final goodFeatureColIdx = 0;
      final ignoredFeatureColIdx = 1;
      final bestFeatureColIdx = 2;

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

      final featuresIdxs = {goodFeatureColIdx, bestFeatureColIdx};

      final assessor = SplitAssessorMock();
      final splitter = DecisionTreeSplitterMock();
      final categoricalValues = [1.0, 2.0, 3.0];
      final colIdxToUniqueValues = {ignoredFeatureColIdx: categoricalValues};

      when(splitter.split(observations, goodFeatureColIdx, targetColIdx))
          .thenReturn(goodSplit);
      when(splitter.split(observations, bestFeatureColIdx, targetColIdx))
          .thenReturn(bestSplit);

      when(assessor.getAggregatedError(goodSplit.values,
          targetColIdx)).thenReturn(0.51);
      when(assessor.getAggregatedError(bestSplit.values,
          targetColIdx)).thenReturn(0.1);

      final selector = GreedySplitSelector(assessor, splitter);
      final stump = selector.select(observations, targetColIdx,
          featuresIdxs, colIdxToUniqueValues);

      expect(stump.keys, equals(bestSplit.keys));
      expect(stump.values, equals(bestSplit.values));

      verifyNever(splitter.split(observations, ignoredFeatureColIdx,
          targetColIdx));
    });
  });
}
