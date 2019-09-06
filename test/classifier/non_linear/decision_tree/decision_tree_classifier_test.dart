import 'package:ml_algo/src/classifier/non_linear/decision_tree/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/non_linear/decision_tree/decision_tree_classifier_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

void main() {
  group('DecisionTreeClassifier', () {
    final featuresForPrediction = Matrix.fromList([
      [200,  300, 1, 0, 0, 10, -40],
      [190, -500, 1, 0, 0, 11, -31],
      [2563, 16,  0, 0, 1, 22,  50],
      [5598, 14,  0, 1, 0, 99, 100],
    ]);

    final dataFrame = DataFrame.fromSeries([
      Series('col_1', <int>[10, 90, 23, 55]),
      Series('col_2', <int>[20, 51, 40, 10]),
      Series('col_3', <int>[1, 0, 0, 1], isDiscrete: true),
      Series('col_4', <int>[0, 0, 1, 0], isDiscrete: true),
      Series('col_5', <int>[0, 1, 0, 0], isDiscrete: true),
      Series('col_6', <int>[30, 34, 90, 22]),
      Series('col_7', <int>[40, 31, 50, 80]),
      Series('col_8', <int>[0, 0, 1, 2], isDiscrete: true),
    ]);

    group('greedy', () {
      final classifier = DecisionTreeClassifier(dataFrame, targetId: 7,
          minError: 0.3, minSamplesCount: 1, maxDepth: 3);

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
              [0],
              [2],
              [0],
              [2],
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
