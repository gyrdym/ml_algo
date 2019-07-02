import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector_impl.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

import '../../../../test_utils/mocks.dart';

void main() {
  group('LeafDetectorImpl', () {
    test('should detect tree leaf if given features column ranges collection '
        'is empty', () {
      final detector = LeafDetectorImpl(null, null, 2);
      final isLeaf = detector.isLeaf(Matrix.fromList([
        [1, 2, 3],
        [2, 3, 4],
        [2, 3, 4],
        [2, 3, 4],
      ]), null, []);

      expect(isLeaf, isTrue);
    });

    test('should detect tree leaf if given samples number is equal to minimum '
        'allowed samples number', () {
      final detector = LeafDetectorImpl(null, null, 2);
      final isLeaf = detector.isLeaf(Matrix.fromList([
        [1, 2, 3],
        [2, 3, 4],
      ]), null, [ZRange.all()]);

      expect(isLeaf, isTrue);
    });

    test('should detect tree leaf if given samples number is less than the '
        'minimum allowed number', () {
      final detector = LeafDetectorImpl(null, null, 2);
      final isLeaf = detector.isLeaf(Matrix.fromList([
        [1, 2, 3],
      ]), null, [ZRange.all()]);

      expect(isLeaf, isTrue);
    });

    test('should detect tree leaf if all labels on node belong to one '
        'class', () {
      final observations = Matrix.fromList([
        [10, 20, 1, 0, 0],
        [10, 20, 1, 0, 0],
        [10, 20, 1, 0, 0],
      ]);
      final detector = LeafDetectorImpl(null, null, 2);
      final isLeaf = detector.isLeaf(observations, ZRange.closed(2, 4),
          [ZRange.all()]);

      expect(isLeaf, isTrue);
    });

    test('should detect tree leaf if mnimum error reached', () {
      final observations = Matrix.fromList([
        [10, 2, 1, 0, 0],
        [20, 3, 0, 1, 0],
      ]);
      final assessor = SplitAssessorMock();
      final detector = LeafDetectorImpl(assessor, 3, 1);

      when(assessor.getError(observations, ZRange.closed(2, 4)))
          .thenReturn(3);

      final isLeaf = detector.isLeaf(observations, ZRange.closed(2, 4),
          [ZRange.all()]);

      expect(isLeaf, isTrue);
    });

    test('should detect tree leaf if current error is less than minimum '
        'error', () {
      final observations = Matrix.fromList([
        [10, 30, 1, 0, 0],
        [40, 50, 0, 1, 0],
      ]);
      final assessor = SplitAssessorMock();
      final detector = LeafDetectorImpl(assessor, 3, 1);

      when(assessor.getError(observations, ZRange.closed(2, 4)))
          .thenReturn(2);

      final isLeaf = detector.isLeaf(observations, ZRange.closed(2, 4),
          [ZRange.all()]);

      expect(isLeaf, isTrue);
    });

    test('should not detect tree leaf if node count have not reached the limit,'
        'all class labels on node do not belong to one class and error on node'
        'is greater than minimum error', () {
      final observations = Matrix.fromList([
        [10, 1, 0, 0],
        [20, 0, 1, 0],
      ]);
      final assessor = SplitAssessorMock();
      final detector = LeafDetectorImpl(assessor, 3, 1);

      when(assessor.getError(observations, ZRange.closed(1, 3)))
          .thenReturn(4);

      final isLeaf = detector.isLeaf(observations, ZRange.closed(1, 3),
          [ZRange.all()]);

      expect(isLeaf, isFalse);
    });
  });
}
