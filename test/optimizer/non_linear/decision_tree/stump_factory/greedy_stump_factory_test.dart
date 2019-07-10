import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_stump.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/nominal_splitter/nominal_splitter.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/numerical_splitter/numerical_splitter.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_factory/greedy_splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

import '../../../../test_utils/mocks.dart';

void main() {
  group('GreedyStumpFactory', () {
    group('(splitting by real number)', () {
      test('should sort observations (ASC-direction) by given column and '
          'create stump with minimal error on split', () {
        final inputObservations = Matrix.fromList([
          [10.0, 5.0],
          [-10.0, -20.0],
          [5.0, 20.0],
          [4.0, 34.0],
          [0.0, 10.0],
        ]);

        final outcomesRange = ZRange.singleton(1);

        final bestSplittingValue = 4.5;
        final splittingColumn = 0;

        final mockedWorstStump = DecisionTreeStump(null, null, null, [
          Matrix.fromList([[12.0, 22.0]]),
          Matrix.fromList([[19.0, 31.0]]),
        ]);

        final mockedWorseStump = DecisionTreeStump(null, null, null, [
          Matrix.fromList([[13.0, 24.0]]),
          Matrix.fromList([[29.0, 53.0]]),
        ]);

        final mockedGoodStump = DecisionTreeStump(null, null, null, [
          Matrix.fromList([[1.0, 2.0]]),
          Matrix.fromList([[9.0, 3.0]]),
        ]);

        final mockedBestStump = DecisionTreeStump(null, null, null, [
          Matrix.fromList([[100.0, 200.0]]),
          Matrix.fromList([[300.0, 400.0]]),
        ]);

        final mockedSplitDataToBeReturned = [
          {
            'splittingValue': -5.0,
            'stump': mockedGoodStump,
          },
          {
            'splittingValue': 2.0,
            'stump': mockedWorseStump,
          },
          {
            'splittingValue': bestSplittingValue,
            'stump': mockedBestStump,
          },
          {
            'splittingValue': 7.5,
            'stump': mockedWorstStump,
          },
        ];

        final assessor = SplitAssessorMock();

        when(assessor.getAggregatedError(mockedWorstStump.outputSamples,
            outcomesRange)).thenReturn(0.99);
        when(assessor.getAggregatedError(mockedWorseStump.outputSamples,
            outcomesRange)).thenReturn(0.8);
        when(assessor.getAggregatedError(mockedGoodStump.outputSamples,
            outcomesRange)).thenReturn(0.4);
        when(assessor.getAggregatedError(mockedBestStump.outputSamples,
            outcomesRange)).thenReturn(0.1);

        final splitter = createNumericalSplitter(mockedSplitDataToBeReturned);
        final stumpFactory = GreedySplitFactory(assessor, splitter, null);
        final stump = stumpFactory.create(inputObservations,
            ZRange.singleton(splittingColumn), outcomesRange);

        for (final splitInfo in mockedSplitDataToBeReturned) {
          final splittingValue = splitInfo['splittingValue'] as double;
          verify(splitter.split(inputObservations, splittingColumn,
              splittingValue)).called(1);
        }

        expect(stump.outputSamples,
            equals(mockedBestStump.outputSamples));
      });
    });

    group('(splitting by categorical values)', () {
      test('should select stump, splitting the observations into parts by '
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
        final splitter = createNominalSplitter(
          splittingValues, [
            Matrix.fromList([
              [11, 22, 0, 0, 1, 30],
              [60, 23, 0, 0, 1, 20],
            ]),
            Matrix.fromList([
              [13, 99, 0, 1, 0, 30],
            ]),
            Matrix.fromList([
              [20, 25, 1, 0, 0, 10],
              [17, 66, 1, 0, 0, 70],
            ]),
          ],
        );
        final selector = GreedySplitFactory(null, null, splitter);
        final stump = selector.create(
          samples,
          splittingColumnRange,
          null,
          splittingValues,
        );

        expect(stump.outputSamples, equals([
          [
            [11, 22, 0, 0, 1, 30],
            [60, 23, 0, 0, 1, 20],
          ],
          [
            [13, 99, 0, 1, 0, 30],
          ],
          [
            [20, 25, 1, 0, 0, 10],
            [17, 66, 1, 0, 0, 70],
          ],
        ]));

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

        final splitter = NominalSplitterMock();
        when(splitter.split(any, any, any)).thenReturn([]);

        final stumpFactory = GreedySplitFactory(null, null, splitter);

        final stump = stumpFactory.create(
          samples,
          splittingColumnRange,
          null,
          splittingValues,
        );
        expect(stump.outputSamples, equals(<Matrix>[]));
        verify(splitter.split(samples, splittingColumnRange, splittingValues))
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
        final stumpFactory = GreedySplitFactory(null, null, null);
        final actual = () => stumpFactory.create(
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
        final selector = GreedySplitFactory(null, null, null);
        final actual = () => selector.create(
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
  final splitter = SamplesByNumericalValueSplitterMock();
  for (final splitInfo in mockedData) {
    final splittingValue = splitInfo['splittingValue'] as double;
    when(splitter.split(any, any, splittingValue)).thenAnswer((_) {
      final stump = splitInfo['stump'] as DecisionTreeStump;
      return stump.outputSamples.toList();
    });
  }
  return splitter;
}

NominalSplitter createNominalSplitter(List<Vector> nominalValues,
    List<Matrix> stumpSamples) {
  final splitter = NominalSplitterMock();
  when(splitter.split(any, any, nominalValues)).thenReturn(stumpSamples);
  return splitter;
}
