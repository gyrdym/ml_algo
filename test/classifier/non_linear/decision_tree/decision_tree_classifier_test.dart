import 'package:ml_algo/src/classifier/non_linear/decision_tree/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/non_linear/decision_tree/decision_tree_classifier_impl.dart';
import 'package:ml_linalg/linalg.dart';
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

    final featuresForPrediction = Matrix.fromList([
      [200,  300, 1, 0, 0, 10, -40],
      [190, -500, 1, 0, 0, 11, -31],
      [2563, 16,  0, 0, 1, 22,  50],
      [5598, 14,  0, 1, 0, 99, 100],
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

    final dataSet = DataSet(observations, outcomesColumnRange,
        rangeToNominalValues);

    group('greedy', () {
      final classifier = DecisionTreeClassifier.greedy(dataSet, 0.3, 1, 3);

      /*
       *          The tree structure:
       *
       *                 (root)
       *                *     *
       *            *             *
       *         *                    *
       *  index: 1                 index: 1
       *  condition: <15           condition: >=15
       *  prediction: 1,0,0           * * *
       *                          *     *      *
       *                      *         *           *
       *                  *             *                *
       *    index: (2-4)         index: (2-4)            index: (2-4)
       *    condition: == 0,0,1  condition: == 0,1,0     condition: == 1,0,0
       *    prediction: 0,0,1    prediction: 0,1,0       prediction: 0,0,1
       */

      test('should create classifier', () {
        expect(classifier, isA<DecisionTreeClassifierImpl>());
        expect(classifier.classLabels, isNull);
      });

      test('should predict class labels', () {
        expect(classifier.predictClasses(featuresForPrediction),
            equals([
              [0, 0, 1],
              [1, 0, 0],
              [0, 0, 1],
              [1, 0, 0],
            ]));
      });

      test('should predict probability of classes', () {
        expect(classifier.predictProbabilities(featuresForPrediction), equals([
              [1],
              [1],
              [1],
              [1],
            ]));
      });
    });
  });
}
