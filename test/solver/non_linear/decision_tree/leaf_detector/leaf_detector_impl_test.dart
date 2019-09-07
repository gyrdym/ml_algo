import 'package:ml_algo/src/solver/non_linear/decision_tree/leaf_detector/leaf_detector_impl.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:quiver/iterables.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

import '../../../../test_utils/mocks.dart';

void main() {
  group('LeafDetectorImpl', () {
    final mockedNodeError = 0.4;
    final minError = 0.3;
    final minSamplesCount = 2;
    final maxDepth = 4;

    test('should detect tree leaf if given depth is greater than the maximum '
        'allowed tree depth', () {
      final assessor = SplitAssessorMock();
      when(assessor.getError(any, any)).thenReturn(mockedNodeError);

      final detector = LeafDetectorImpl(assessor, minError, minSamplesCount,
          maxDepth);

      final isLeaf = detector.isLeaf(Matrix.fromList([
        [1, 2, 3],
        [2, 3, 4],
        [2, 3, 4],
        [2, 3, 4],
      ]), null, count(0).take(2), 10);

      expect(isLeaf, isTrue);
    });

    test('should detect tree leaf if given depth is equal to the maximum '
        'allowed tree depth value', () {
      final assessor = SplitAssessorMock();
      when(assessor.getError(any, any)).thenReturn(mockedNodeError);

      final detector = LeafDetectorImpl(assessor, minError, minSamplesCount,
          maxDepth);

      final isLeaf = detector.isLeaf(Matrix.fromList([
        [1, 2, 3],
        [2, 3, 4],
        [2, 3, 4],
        [2, 3, 4],
      ]), null, count(0).take(2), 4);

      expect(isLeaf, isTrue);
    });

    test('should detect tree leaf if given features column ranges collection '
        'is empty', () {
      final assessor = SplitAssessorMock();
      when(assessor.getError(any, any)).thenReturn(mockedNodeError);

      final detector = LeafDetectorImpl(assessor, minError, minSamplesCount,
          maxDepth);

      final isLeaf = detector.isLeaf(Matrix.fromList([
        [1, 2, 3],
        [2, 3, 4],
        [2, 3, 4],
        [2, 3, 4],
      ]), null, [], 0);

      expect(isLeaf, isTrue);
    });

    test('should detect tree leaf if given samples number is equal to minimum '
        'allowed samples number', () {
      final assessor = SplitAssessorMock();
      when(assessor.getError(any, any)).thenReturn(mockedNodeError);

      final detector = LeafDetectorImpl(assessor, minError, minSamplesCount,
          maxDepth);

      final isLeaf = detector.isLeaf(Matrix.fromList([
        [1, 2, 3],
        [2, 3, 4],
      ]), null, count(0).take(2), 0);

      expect(isLeaf, isTrue);
    });

    test('should detect tree leaf if given samples number is less than the '
        'minimum allowed number', () {
      final assessor = SplitAssessorMock();
      when(assessor.getError(any, any)).thenReturn(mockedNodeError);

      final detector = LeafDetectorImpl(assessor, minError, minSamplesCount,
          maxDepth);

      final isLeaf = detector.isLeaf(Matrix.fromList([
        [1, 2, 3],
      ]), null, count(0).take(2), 0);

      expect(isLeaf, isTrue);
    });

    test('should detect tree leaf if all labels on node belong to one '
        'class', () {
      final observations = Matrix.fromList([
        [10, 20, 1],
        [10, 20, 1],
        [10, 20, 1],
      ]);

      final assessor = SplitAssessorMock();
      when(assessor.getError(any, any)).thenReturn(mockedNodeError);

      final detector = LeafDetectorImpl(assessor, minError, minSamplesCount,
          maxDepth);

      final isLeaf = detector.isLeaf(observations, 2, count(0).take(2), 0);

      expect(isLeaf, isTrue);
    });

    test('should detect tree leaf if mnimum error reached', () {
      final observations = Matrix.fromList([
        [10, 2, 1],
        [20, 3, 2],
        [20, 3, 2],
        [20, 3, 1],
      ]);
      final assessor = SplitAssessorMock();
      when(assessor.getError(any, any)).thenReturn(0.3);

      final detector = LeafDetectorImpl(assessor, minError, minSamplesCount,
          maxDepth);

      final isLeaf = detector.isLeaf(observations, 2, count(0).take(2), 0);

      expect(isLeaf, isTrue);
    });

    test('should detect tree leaf if current error is less than minimum '
        'error', () {
      final observations = Matrix.fromList([
        [10, 30, 1],
        [40, 50, 2],
        [40, 50, 2],
        [40, 50, 1],
      ]);

      final assessor = SplitAssessorMock();
      when(assessor.getError(any, any)).thenReturn(0.1);

      final detector = LeafDetectorImpl(assessor, minError, minSamplesCount,
          maxDepth);

      final isLeaf = detector.isLeaf(observations, 2, count(0).take(2), 0);

      expect(isLeaf, isTrue);
    });

    test('should not detect tree leaf if node count have not reached the limit,'
        'all class labels on node do not belong to one class and error on node'
        'is greater than minimum error', () {
      final observations = Matrix.fromList([
        [10, 1],
        [20, 2],
        [20, 3],
        [20, 1],
      ]);

      final assessor = SplitAssessorMock();
      when(assessor.getError(any, any)).thenReturn(0.5);

      final detector = LeafDetectorImpl(assessor, minError, minSamplesCount,
          maxDepth);

      final isLeaf = detector.isLeaf(observations, 1, count(0).take(1), 0);

      expect(isLeaf, isFalse);
    });
  });
}