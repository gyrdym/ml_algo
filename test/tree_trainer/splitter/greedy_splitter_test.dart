import 'package:ml_algo/src/tree_trainer/splitter/greedy_splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter.dart';
import 'package:ml_algo/src/tree_trainer/splitter/numerical_splitter/numerical_splitter.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.mocks.dart';

void main() {
  final treeSplitAssessorMock = MockTreeSplitAssessor();
  final numericalTreeSplitterMock = MockNumericalTreeSplitter();
  final nominalTreeSplitterMock = MockNominalTreeSplitter();

  group('GreedyTreeSplitter', () {
    test(
        'should sort observations (ASC-direction) by given column and '
        'create split with minimal error if split by continuous value is '
        'happening', () {
      final inputObservations = Matrix.fromList([
        [10.0, 5.0],
        [-10.0, -20.0],
        [5.0, 20.0],
        [4.0, 34.0],
        [0.0, 10.0],
      ]);
      final targetIdx = 1;
      final bestSplittingValue = 4.5;
      final splittingColumnIdx = 0;
      final mockedWorstSplit = {
        MockTreeNode(): Matrix.fromList([
          [12.0, 22.0]
        ]),
        MockTreeNode(): Matrix.fromList([
          [19.0, 31.0]
        ]),
      };
      final mockedWorseSplit = {
        MockTreeNode(): Matrix.fromList([
          [13.0, 24.0]
        ]),
        MockTreeNode(): Matrix.fromList([
          [29.0, 53.0]
        ]),
      };
      final mockedGoodSplit = {
        MockTreeNode(): Matrix.fromList([
          [1.0, 2.0]
        ]),
        MockTreeNode(): Matrix.fromList([
          [9.0, 3.0]
        ]),
      };
      final bestNodeLeft = MockTreeNode();
      final bestNodeRight = MockTreeNode();
      final mockedBestSplit = {
        bestNodeLeft: Matrix.fromList([
          [100.0, 200.0]
        ]),
        bestNodeRight: Matrix.fromList([
          [300.0, 400.0]
        ]),
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
      final assessor = MockTreeSplitAssessor();

      when(assessor.getAggregatedError(mockedWorstSplit.values, targetIdx))
          .thenReturn(0.99);
      when(assessor.getAggregatedError(mockedWorseSplit.values, targetIdx))
          .thenReturn(0.8);
      when(assessor.getAggregatedError(mockedGoodSplit.values, targetIdx))
          .thenReturn(0.4);
      when(assessor.getAggregatedError(mockedBestSplit.values, targetIdx))
          .thenReturn(0.1);

      final numericalSplitter =
          createNumericalSplitter(mockedSplitDataToBeReturned);
      final splitter = GreedyTreeSplitter(
        assessor,
        numericalSplitter,
        nominalTreeSplitterMock,
      );
      final actualSplit =
          splitter.split(inputObservations, splittingColumnIdx, targetIdx);

      for (final splitInfo in mockedSplitDataToBeReturned) {
        final splittingValue = splitInfo['splittingValue'] as double;

        verify(
          numericalSplitter.split(
            inputObservations,
            splittingColumnIdx,
            splittingValue,
          ),
        ).called(1);
      }

      expect(actualSplit.keys, equals(mockedBestSplit.keys));
      expect(actualSplit.values, equals(mockedBestSplit.values));
    });
  });

  test(
      'should create split, dividing the observations into parts by '
      'given splitting column index', () {
    final samples = Matrix.fromList([
      [11, 22, 1, 30],
      [60, 23, 1, 20],
      [20, 25, 2, 10],
      [17, 66, 2, 70],
      [13, 99, 3, 30],
    ]);
    final splittingColumnIdx = 2;
    final splittingValues = [1.0, 3.0, 2.0];
    final mockedSplit = {
      MockTreeNode(): Matrix.fromList([
        [11, 22, 1, 30],
        [60, 23, 1, 20],
      ]),
      MockTreeNode(): Matrix.fromList([
        [13, 99, 3, 30],
      ]),
      MockTreeNode(): Matrix.fromList([
        [20, 25, 2, 10],
        [17, 66, 2, 70],
      ]),
    };
    final splitter = createNominalSplitter(splittingValues, mockedSplit);
    final selector = GreedyTreeSplitter(
      treeSplitAssessorMock,
      numericalTreeSplitterMock,
      splitter,
    );
    final actualSplit = selector.split(
      samples,
      splittingColumnIdx,
      -1,
      splittingValues,
    );

    expect(actualSplit.keys, equals(mockedSplit.keys));
    expect(actualSplit.values, equals(mockedSplit.values));

    verify(splitter.split(samples, splittingColumnIdx, splittingValues))
        .called(1);
  });

  test(
      'should return an empty stum if splitting values collection is '
      'empty', () {
    final samples = Matrix.fromList([
      [11, 22, 1, 30],
      [60, 23, 1, 20],
      [20, 25, 2, 10],
      [17, 66, 2, 70],
      [13, 99, 3, 30],
    ]);
    final splittingColumnIdx = 2;
    final splittingValues = <double>[];
    final nominalSplitter = MockNominalTreeSplitter();

    when(nominalSplitter.split(
      any,
      any,
      any,
    )).thenReturn({});

    final splitter = GreedyTreeSplitter(
      treeSplitAssessorMock,
      numericalTreeSplitterMock,
      nominalSplitter,
    );

    final split = splitter.split(
      samples,
      splittingColumnIdx,
      -1,
      splittingValues,
    );
    expect(split.values, equals(<Matrix>[]));
    verify(nominalSplitter.split(samples, splittingColumnIdx, splittingValues))
        .called(1);
  });

  test(
      'should skip the same values while iterating through the sorted '
      'rows', () {
    final samples = Matrix.fromList([
      [11],
      [11],
      [11],
      [20],
    ]);
    final splittingColumnIdx = 0;
    final targetColumnIdx = 1;

    final numericalSplitter = MockNumericalTreeSplitter();
    final assessor = MockTreeSplitAssessor();

    when(
      numericalSplitter.split(
        samples,
        splittingColumnIdx,
        any,
      ),
    ).thenReturn({});
    when(
      assessor.getError(
        any,
        targetColumnIdx,
      ),
    ).thenReturn(0.5);
    when(
      assessor.getAggregatedError(
        any,
        any,
      ),
    ).thenReturn(1.0);

    final splitter = GreedyTreeSplitter(
      assessor,
      numericalSplitter,
      nominalTreeSplitterMock,
    );

    final split = splitter.split(
      samples,
      splittingColumnIdx,
      targetColumnIdx,
      null,
    );
    expect(split.values, equals(<Matrix>[]));

    final verificationResult = verify(
      numericalSplitter.split(
        samples,
        splittingColumnIdx,
        captureThat(isNotNull),
      ),
    );

    verificationResult.called(1);

    expect(verificationResult.captured[0], 15.5);
  });

  test('should process split column with the same values', () {
    final samples = Matrix.fromList([
      [11],
      [11],
      [11],
      [11],
    ]);
    final splittingColumnIdx = 0;
    final targetColumnIdx = 1;

    final numericalSplitter = MockNumericalTreeSplitter();
    final assessor = MockTreeSplitAssessor();

    when(
      numericalSplitter.split(
        samples,
        splittingColumnIdx,
        any,
      ),
    ).thenReturn({});
    when(
      assessor.getError(
        any,
        targetColumnIdx,
      ),
    ).thenReturn(0.5);
    when(
      assessor.getAggregatedError(
        any,
        any,
      ),
    ).thenReturn(1.0);

    final splitter = GreedyTreeSplitter(
      assessor,
      numericalSplitter,
      nominalTreeSplitterMock,
    );

    final split = splitter.split(
      samples,
      splittingColumnIdx,
      targetColumnIdx,
      null,
    );
    expect(split.values, equals(<Matrix>[]));

    final verificationResult = verify(
      numericalSplitter.split(
        samples,
        splittingColumnIdx,
        captureThat(isNotNull),
      ),
    );

    verificationResult.called(1);

    expect(verificationResult.captured[0], 11.0);
  });

  test('should throw an error if negative splitting index is given', () {
    final samples = Matrix.fromList([
      [11, 22, 1, 30],
      [60, 23, 1, 20],
      [20, 25, 2, 10],
      [17, 66, 2, 70],
      [13, 99, 3, 30],
    ]);
    final splittingColumnIdx = -2;
    final splittingValues = [1.0, 3.0];
    final splitter = GreedyTreeSplitter(
      treeSplitAssessorMock,
      numericalTreeSplitterMock,
      nominalTreeSplitterMock,
    );
    final actual = () => splitter.split(
          samples,
          splittingColumnIdx,
          -1,
          splittingValues,
        );
    expect(actual, throwsException);
  });

  test(
      'should throw an error if splitting index is greater than the '
      'number of columns', () {
    final samples = Matrix.fromList([
      [11, 22, 1, 30],
      [60, 23, 1, 20],
      [20, 25, 2, 10],
      [17, 66, 2, 70],
      [13, 99, 3, 30],
    ]);
    final splittingColumnIdx = 10;
    final splittingValues = [1.0, 3.0];
    final selector = GreedyTreeSplitter(
      treeSplitAssessorMock,
      numericalTreeSplitterMock,
      nominalTreeSplitterMock,
    );
    final actual = () => selector.split(
          samples,
          splittingColumnIdx,
          -1,
          splittingValues,
        );

    expect(actual, throwsException);
  });
}

NumericalTreeSplitter createNumericalSplitter(
    List<Map<String, dynamic>> mockedData) {
  final splitter = MockNumericalTreeSplitter();

  for (final splitInfo in mockedData) {
    final splittingValue = splitInfo['splittingValue'] as double;

    when(
      splitter.split(
        any,
        any,
        splittingValue,
      ),
    ).thenAnswer((_) => splitInfo['split'] as Map<TreeNode, Matrix>);
  }

  return splitter;
}

NominalTreeSplitter createNominalSplitter(
    List<double> nominalValues, Map<TreeNode, Matrix> split) {
  final splitter = MockNominalTreeSplitter();

  when(splitter.split(
    any,
    any,
    nominalValues,
  )).thenReturn(split);

  return splitter;
}
