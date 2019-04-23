import 'package:ml_algo/src/model_selection/data_splitter/leave_p_out.dart';
import 'package:test/test.dart';

void main() {
  group('Leave p out splitter', () {
    void testLpoSplitter(int p, int numOfObservations,
        Iterable<Iterable<int>> expected) {
      test('should return proper groups of indices if p is $p and number of '
          'observations is $numOfObservations', () {
        final splitter = LeavePOutSplitter(p);
        expect(splitter.split(numOfObservations).toSet(), equals(expected));
      });
    }

    testLpoSplitter(2, 4, [
      [0, 1],
      [0, 2],
      [0, 3],
      [1, 2],
      [1, 3],
      [2, 3],
    ].toSet());

    testLpoSplitter(2, 5, [
      [0, 1],
      [0, 2],
      [0, 3],
      [0, 4],
      [1, 2],
      [1, 3],
      [1, 4],
      [2, 3],
      [2, 4],
      [3, 4],
    ].toSet());

    testLpoSplitter(1, 5, [
      [0],
      [1],
      [2],
      [3],
      [4],
    ].toSet());

    testLpoSplitter(3, 4, [
      [0, 1, 2],
      [0, 1, 3],
      [0, 2, 3],
      [1, 2, 3],
    ].toSet());

    testLpoSplitter(3, 5, [
      [0, 1, 2],
      [0, 1, 3],
      [0, 1, 4],
      [0, 2, 3],
      [0, 2, 4],
      [0, 3, 4],
      [1, 2, 3],
      [1, 2, 4],
      [1, 3, 4],
      [2, 3, 4],
    ].toSet());

    test('should throw an error, if p is equal to 0', () {
      expect(() => LeavePOutSplitter(0), throwsUnsupportedError);
    });
  });
}
