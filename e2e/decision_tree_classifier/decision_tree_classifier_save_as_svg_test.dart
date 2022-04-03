import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:test/test.dart';

void main() async {
  group('DecisionTreeClassifier', () {
    test('should save graphical representation as svg image', () async {
      final samples = (await fromCsv('e2e/_datasets/iris.csv'))
          .shuffle()
          .dropSeries(seriesNames: ['Id']);
      final pipeline = Pipeline(samples, [
        encodeAsIntegerLabels(
          featureNames: ['Species'],
        ),
      ]);
      final processed = pipeline.process(samples);
      final classifier = DecisionTreeClassifier(
        processed,
        'Species',
        minError: 0.3,
        minSamplesCount: 5,
        maxDepth: 4,
      );

      await classifier.saveAsSvg('e2e/decision_tree_classifier/iris_tree.svg');
    });

    test('should save graphical representation as svg image', () async {
      final samples =
          (await fromCsv('e2e/_datasets/pima_indians_diabetes_database.csv'))
              .shuffle();
      final classifier = DecisionTreeClassifier(
        samples,
        'class variable (0 or 1)',
        minError: 0.1,
        minSamplesCount: 2,
        maxDepth: 4,
      );

      await classifier
          .saveAsSvg('e2e/decision_tree_classifier/pima_indians_tree.svg');
    });
  });
}
