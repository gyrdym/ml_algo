import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_selector/number_based/greedy_number_based_stump_selector.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_selector/number_based/node_splitter/node_splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../../../../test_utils/mocks.dart';

void main() {
  group('GreedyNumberBasedStumpSelector', () {
    test('should sort observations (ASC-direction) by given column and select '
        'stump with minimal error', () {
      final inputObservations = Matrix.fromList([
        [10.0, 5.0],
        [-10.0, -20.0],
        [5.0, 20.0],
        [4.0, 34.0],
        [0.0, 10.0],
      ]);

      final bestSplittingValue = 4.5;
      final splittingColumn = 0;

      final mockedWorstStump = [
        [[12.0, 22.0]],
        [[19.0, 31.0]],
      ];

      final mockedWorseStump = [
        [[13.0, 24.0]],
        [[29.0, 53.0]],
      ];

      final mockedGoodStump = [
        [[1.0, 2.0]],
        [[9.0, 3.0]],
      ];

      final mockedBestStump = [
        [[100.0, 200.0]],
        [[300.0, 400.0]],
      ];

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

      final assessor = StumpAssessorMock();

      when(assessor.getErrorOnStump(argThat(equals(mockedWorstStump))))
          .thenReturn(0.99);
      when(assessor.getErrorOnStump(argThat(equals(mockedWorseStump))))
          .thenReturn(0.8);
      when(assessor.getErrorOnStump(argThat(equals(mockedGoodStump))))
          .thenReturn(0.4);
      when(assessor.getErrorOnStump(argThat(equals(mockedBestStump))))
          .thenReturn(0.1);

      final splitter = createSplitter(mockedSplitDataToBeReturned);
      final selector = GreedyNumberBasedStumpSelector(assessor, splitter);
      final nodes = selector.select(inputObservations, splittingColumn);

      for (final splitInfo in mockedSplitDataToBeReturned) {
        final splittingValue = splitInfo['splittingValue'] as double;
        verify(splitter.split(argThat(equals(inputObservations)),
            splittingColumn, splittingValue)).called(1);
      }

      expect(nodes, equals(mockedBestStump));
    });
  });
}

NodeSplitter createSplitter(List<Map<String, dynamic>> mockedData) {
  final splitter = NodeSplitterMock();
  for (final splitInfo in mockedData) {
    final splittingValue = splitInfo['splittingValue'] as double;
    when(splitter.split(any, any, splittingValue)).thenAnswer((_) {
      final stump = splitInfo['stump'] as List<List<List<double>>>;
      return List.generate(stump.length, (i) => Matrix.fromList(stump[i]));
    });
  }
  return splitter;
}
