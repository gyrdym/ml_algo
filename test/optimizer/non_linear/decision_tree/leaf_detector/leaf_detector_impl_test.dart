import 'package:ml_algo/src/optimizer/non_linear/decision_tree/assessor/stump_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_detector/leaf_detector_impl.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../../../test_utils/mocks.dart';

void main() {
  group('LeafDetectorImpl', () {
    test('should detect tree leaf if maximum node count was reached', () {
      final detector = const LeafDetectorImpl(null, null, 10);
      final isLeaf = detector.isLeaf(null, 10);
      expect(isLeaf, isTrue);
    });

    test('should detect tree leaf if current node count exceeded the '
        'limit', () {
      final detector = const LeafDetectorImpl(null, null, 10);
      final isLeaf = detector.isLeaf(null, 11);
      expect(isLeaf, isTrue);
    });

    test('should detect tree leaf if all labels on node belong to one '
        'class', () {
      final outcomes = Matrix.fromList([
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
      ]);
      final detector = const LeafDetectorImpl(null, null, 10);
      final isLeaf = detector.isLeaf(outcomes, 3);
      expect(isLeaf, isTrue);
    });

    test('should detect tree leaf if mnimum error reached', () {
      final outcomes = Matrix.fromList([
        [1, 0, 0],
        [0, 1, 0],
      ]);
      final assessor = StumpAssessorMock();
      final detector = LeafDetectorImpl(assessor, 3, 10);

      when(assessor.getErrorOnNode(outcomes)).thenReturn(3);

      final isLeaf = detector.isLeaf(outcomes, 3);
      expect(isLeaf, isTrue);
    });

    test('should detect tree leaf if current error is less than minimum '
        'error', () {
      final outcomes = Matrix.fromList([
        [1, 0, 0],
        [0, 1, 0],
      ]);
      final assessor = StumpAssessorMock();
      final detector = LeafDetectorImpl(assessor, 3, 10);

      when(assessor.getErrorOnNode(outcomes)).thenReturn(2);

      final isLeaf = detector.isLeaf(outcomes, 3);
      expect(isLeaf, isTrue);
    });

    test('should not detect tree leaf if node count have not reached the limit,'
        'all class labels on node do not belong to one class and error on node'
        'is greater than minimum error', () {
      final outcomes = Matrix.fromList([
        [1, 0, 0],
        [0, 1, 0],
      ]);
      final assessor = StumpAssessorMock();
      final detector = LeafDetectorImpl(assessor, 3, 10);

      when(assessor.getErrorOnNode(outcomes)).thenReturn(4);

      final isLeaf = detector.isLeaf(outcomes, 3);
      expect(isLeaf, isFalse);
    });
  });
}
