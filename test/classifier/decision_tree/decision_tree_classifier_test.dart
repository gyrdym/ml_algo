import 'package:ml_algo/src/classifier/decision_tree/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree/decision_tree_classifier_impl.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:test/test.dart';
import 'package:xrange/zrange.dart';

void main() {
  group('DecisionTreeClassifier', () {
    final observations = Matrix.fromList([
      [10, 20, 1, 0, 0, 30, 40, 0, 0, 1],
      [90, 51, 0, 0, 1, 34, 31, 0, 0, 1],
      [23, 40, 0, 1, 0, 90, 50, 0, 1, 0],
      [55, 10, 1, 0, 0, 22, 80, 1, 0, 0],
    ]);

    final nominalFeatureRange = ZRange.closed(2, 4);
    final outcomesColumnRange = ZRange.closed(7, 9);

    final rangeToNominalValues = {
      nominalFeatureRange: [
        // order of nominal features is valuable for building the tree -
        // the earlier value is in the list, the earlier the appropriate
        // node will be built
        Vector.fromList([0, 0, 1]),
        Vector.fromList([0, 1, 0]),
        Vector.fromList([1, 0, 0]),
      ],
      outcomesColumnRange: [
        Vector.fromList([0, 0, 1]),
        Vector.fromList([0, 1, 0]),
        Vector.fromList([1, 0, 0]),
      ],
    };

    final idxToRanges = {
      0: ZRange.singleton(0),
      1: ZRange.singleton(1),
      2: nominalFeatureRange,
      3: nominalFeatureRange,
      4: nominalFeatureRange,
      5: ZRange.singleton(5),
      6: ZRange.singleton(6),
      7: outcomesColumnRange,
      8: outcomesColumnRange,
      9: outcomesColumnRange,
    };

    test('should create greedy decision tree classifier', () {
      final dataSet = DataSet(observations, outcomesColumnRange, idxToRanges,
          rangeToNominalValues);
      final classifier = DecisionTreeClassifier.greedy(dataSet, 0.3, 3, 4);

      expect(classifier, isA<DecisionTreeClassifierImpl>());
    });
  });
}
