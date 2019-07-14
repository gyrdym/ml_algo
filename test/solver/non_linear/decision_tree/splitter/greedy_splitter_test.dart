import 'package:ml_algo/src/solver/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/splitter/greedy_splitter.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/splitter/nominal_splitter/nominal_splitter.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/splitter/numerical_splitter/numerical_splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

import '../../../../test_utils/mocks.dart';

void main() {
  group('GreedySplitter', () {
    group('(splitting by real number)', () {
      test('should sort observations (ASC-direction) by given column and '
          'create split with minimal error', () {
        final inputObservations = Matrix.fromList([
          [10.0, 5.0],
          [-10.0, -20.0],
          [5.0, 20.0],
          [4.0, 34.0],
          [0.0, 10.0],
        ]);

        final outcomesRange = ZRange.singleton(1);

        final bestSplittingValue = 4.5;
        final splittingColumn = ZRange.singleton(0);

        final mockedWorstSplit = {
          DecisionTreeNodeMock(): Matrix.fromList([[12.0, 22.0]]),
          DecisionTreeNodeMock(): Matrix.fromList([[19.0, 31.0]]),
        };

        final mockedWorseSplit = {
          DecisionTreeNodeMock(): Matrix.fromList([[13.0, 24.0]]),
          DecisionTreeNodeMock(): Matrix.fromList([[29.0, 53.0]]),
        };

        final mockedGoodSplit = {
          DecisionTreeNodeMock(): Matrix.fromList([[1.0, 2.0]]),
          DecisionTreeNodeMock(): Matrix.fromList([[9.0, 3.0]]),
        };

        final bestNodeLeft = DecisionTreeNodeMock();
        final bestNodeRight = DecisionTreeNodeMock();

        final mockedBestSplit = {
          bestNodeLeft: Matrix.fromList([[100.0, 200.0]]),
          bestNodeRight: Matrix.fromList([[300.0, 400.0]]),
        };

        final mockedSplitDataToBeReturned = [
          {
            'splittingValue': -5.0,
            'split': mockedGoodSplit,
          },
          {
            'splittingValue': 2.0,
            'split': mockedWorseSplit,
          },
          {
            'splittingValue': bestSplittingValue,
            'split': mockedBestSplit,
          },
          {
            'splittingValue': 7.5,
            'split': mockedWorstSplit,
          },
        ];

        final assessor = SplitAssessorMock();

        when(assessor.getAggregatedError(mockedWorstSplit.values,
            outcomesRange)).thenReturn(0.99);
        when(assessor.getAggregatedError(mockedWorseSplit.values,
            outcomesRange)).thenReturn(0.8);
        when(assessor.getAggregatedError(mockedGoodSplit.values,
            outcomesRange)).thenReturn(0.4);
        when(assessor.getAggregatedError(mockedBestSplit.values,
            outcomesRange)).thenReturn(0.1);

        final numericalSplitter = createNumericalSplitter(
            mockedSplitDataToBeReturned);
        final splitter = GreedySplitter(assessor, numericalSplitter, null);
        final actualSplit = splitter.split(inputObservations,
            splittingColumn, outcomesRange);

        for (final splitInfo in mockedSplitDataToBeReturned) {
          final splittingValue = splitInfo['splittingValue'] as double;
          verify(numericalSplitter.split(inputObservations, splittingColumn,
              splittingValue)).called(1);
        }

        expect(actualSplit.keys, equals(mockedBestSplit.keys));
        expect(actualSplit.values, equals(mockedBestSplit.values));
      });
    });

    group('(splitting by categorical values)', () {
      test('should create split, splitting the observations into parts by '
          'given column range', () {
        final samples = Matrix.fromList([
          [11, 22, 0, 0, 1, 30],
          [60, 23, 0, 0, 1, 20],
          [20, 25, 1, 0, 0, 10],
          [17, 66, 1, 0, 0, 70],
          [13, 99, 0, 1, 0, 30],
        ]);
        final splittingColumnRange = ZRange.closed(2, 4);
        final splittingValues = [
          Vector.fromList([0, 0, 1]),
          Vector.fromList([0, 1, 0]),
          Vector.fromList([1, 0, 0]),
        ];
        final mockedSplit = {
          DecisionTreeNodeMock(): Matrix.fromList([
            [11, 22, 0, 0, 1, 30],
            [60, 23, 0, 0, 1, 20],
          ]),
          DecisionTreeNodeMock(): Matrix.fromList([
            [13, 99, 0, 1, 0, 30],
          ]),
          DecisionTreeNodeMock(): Matrix.fromList([
            [20, 25, 1, 0, 0, 10],
            [17, 66, 1, 0, 0, 70],
          ]),
        };
        final splitter = createNominalSplitter(splittingValues, mockedSplit);
        final selector = GreedySplitter(null, null, splitter);
        final actualSplit = selector.split(
          samples,
          splittingColumnRange,
          null,
          splittingValues,
        );

        expect(actualSplit.keys, equals(mockedSplit.keys));
        expect(actualSplit.values, equals(mockedSplit.values));

        verify(splitter.split(samples, splittingColumnRange,
            splittingValues)).called(1);
      });

      test('should return an empty stum if splitting values collection is '
          'empty', () {
        final samples = Matrix.fromList([
          [11, 22, 0, 0, 1, 30],
          [60, 23, 0, 0, 1, 20],
          [20, 25, 1, 0, 0, 10],
          [17, 66, 1, 0, 0, 70],
          [13, 99, 0, 1, 0, 30],
        ]);
        final splittingColumnRange = ZRange.closed(2, 4);
        final splittingValues = <Vector>[];

        final nominalSplitter = NominalSplitterMock();
        when(nominalSplitter.split(any, any, any)).thenReturn({});

        final splitter = GreedySplitter(null, null, nominalSplitter);

        final split = splitter.split(
          samples,
          splittingColumnRange,
          null,
          splittingValues,
        );
        expect(split.values, equals(<Matrix>[]));
        verify(nominalSplitter.split(samples, splittingColumnRange, splittingValues))
            .called(1);
      });

      test('should throw an error if unappropriate range is given (left '
          'boundary is less than 0)', () {
        final samples = Matrix.fromList([
          [11, 22, 0, 0, 1, 30],
          [60, 23, 0, 0, 1, 20],
          [20, 25, 1, 0, 0, 10],
          [17, 66, 1, 0, 0, 70],
          [13, 99, 0, 1, 0, 30],
        ]);
        final splittingColumnRange = ZRange.closed(-2, 4);
        final splittingValues = [
          Vector.fromList([0, 0, 1]),
          Vector.fromList([0, 1, 0]),
        ];
        final splitter = GreedySplitter(null, null, null);
        final actual = () => splitter.split(
          samples,
          splittingColumnRange,
          null,
          splittingValues,
        );
        expect(actual, throwsException);
      });

      test('should throw an error if unappropriate range is given (right '
          'boundary is greater than the observations columns number)', () {
        final samples = Matrix.fromList([
          [11, 22, 0, 0, 1, 30],
          [60, 23, 0, 0, 1, 20],
          [20, 25, 1, 0, 0, 10],
          [17, 66, 1, 0, 0, 70],
          [13, 99, 0, 1, 0, 30],
        ]);
        final splittingColumnRange = ZRange.closed(0, 10);
        final splittingValues = [
          Vector.fromList([0, 0, 1]),
          Vector.fromList([0, 1, 0]),
        ];
        final selector = GreedySplitter(null, null, null);
        final actual = () => selector.split(
          samples,
          splittingColumnRange,
          null,
          splittingValues,
        );
        expect(actual, throwsException);
      });
    });
  });
}

NumericalSplitter createNumericalSplitter(
    List<Map<String, dynamic>> mockedData) {
  final splitter = NumericalSplitterMock();
  for (final splitInfo in mockedData) {
    final splittingValue = splitInfo['splittingValue'] as double;
    when(splitter.split(any, any, splittingValue)).thenAnswer((_) =>
      splitInfo['split'] as Map<DecisionTreeNode, Matrix>);
  }
  return splitter;
}

NominalSplitter createNominalSplitter(List<Vector> nominalValues,
    Map<DecisionTreeNode, Matrix> split) {
  final splitter = NominalSplitterMock();
  when(splitter.split(any, any, nominalValues)).thenReturn(split);
  return splitter;
}
