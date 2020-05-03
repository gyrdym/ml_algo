import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_impl.dart';
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

    final classifier = DecisionTreeClassifier(dataFrame, 'col_8',
        minError: 0.3, minSamplesCount: 1, maxDepth: 3);

    test('should create classifier', () {
      expect(classifier, isA<DecisionTreeClassifierImpl>());
    });

    test('should throw an exception if target column does not exist in train '
        'data', () {
      final actual = () => DecisionTreeClassifier(dataFrame, 'unknown_col',
          minError: 0.3, minSamplesCount: 1, maxDepth: 3);

      expect(actual, throwsException);
    });

    test('should throw a range error if minimal error on node is less than 0',
            () {
      final actual = () => DecisionTreeClassifier(dataFrame, 'col_8',
          minError: -0.001, minSamplesCount: 1, maxDepth: 3);

      expect(actual, throwsRangeError);
    });

    test('should throw a range error if minimal error on node is greater '
        'than 1', () {
      final actual = () => DecisionTreeClassifier(dataFrame, 'col_8',
          minError: 1.001, minSamplesCount: 1, maxDepth: 3);

      expect(actual, throwsRangeError);
    });

    test('should allow minimal error to be 0', () {
      final actual = DecisionTreeClassifier(dataFrame, 'col_8',
          minError: 0, minSamplesCount: 1, maxDepth: 3);

      expect(actual, isA<DecisionTreeClassifierImpl>());
    });

    test('should allow minimal error to be 1', () {
      final actual = DecisionTreeClassifier(dataFrame, 'col_8',
          minError: 1, minSamplesCount: 1, maxDepth: 3);

      expect(actual, isA<DecisionTreeClassifierImpl>());
    });

    test('should throw an exception if minimal samples count on node is less '
        'than zero', () {
      final actual = () => DecisionTreeClassifier(dataFrame, 'col_8',
          minError: 0.5, minSamplesCount: -1, maxDepth: 3);

      expect(actual, throwsException);
    });

    test('should throw an exception if minimal samples count on node is equal '
        'to zero', () {
      final actual = () => DecisionTreeClassifier(dataFrame, 'col_8',
          minError: 0.5, minSamplesCount: 0, maxDepth: 3);

      expect(actual, throwsException);
    });

    test('should throw an exception if maximal tree depth value is less '
        'than zero', () {
      final actual = () => DecisionTreeClassifier(dataFrame, 'col_8',
          minError: 0.5, minSamplesCount: 1, maxDepth: -13);

      expect(actual, throwsException);
    });

    test('should throw an exception if maximal tree depth value is equal '
        'to zero', () {
      final actual = () => DecisionTreeClassifier(dataFrame, 'col_8',
          minError: 0.5, minSamplesCount: 1, maxDepth: 0);

      expect(actual, throwsException);
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

    test('should throw an error if empty json was passed', () {
      final json = '';
      expect(() => DecisionTreeClassifier.fromJson(json), throwsException);
    });

    test('should throw an error if invalid json was passed', () {
      final json = 'invalid_json';
      expect(() => DecisionTreeClassifier.fromJson(json), throwsException);
    });

    test('should throw an error if passed json has a wrong schema', () {
      final json = '{"field_1": "data"}';
      expect(() => DecisionTreeClassifier.fromJson(json), throwsException);
    });
  });
}
