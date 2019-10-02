import 'package:ml_algo/src/classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier_impl.dart';
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
      final classifier = DecisionTreeClassifier.majority(dataFrame, 'col_8',
          minError: 0.3, minSamplesCount: 1, maxDepth: 3);

      test('should create classifier', () {
        expect(classifier, isA<DecisionTreeClassifierImpl>());
      });

      test('should predict class labels', () {
        final prediction = classifier.predict(
          DataFrame.fromMatrix(featuresForPrediction),
        );

        expect(prediction.header, equals(['col_8']));
        expect(prediction.toMatrix(),
            equals([
              [0],
              [2],
              [0],
              [2],
            ]));
      });

      test('should predict probabilities of classes', () {
        final probabilities = classifier.predictProbabilities(
          DataFrame.fromMatrix(featuresForPrediction),
        );

        expect(probabilities.header, equals(['col_8']));
        expect(probabilities.toMatrix(), equals([
              [1],
              [1],
              [1],
              [1],
            ]));
      });
    });
  });
}
