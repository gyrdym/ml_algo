import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:test/test.dart';

void main() async {
  group('DecisionTreeClassifier', () {
    test('should save graphical representation as svg image, iris dataset',
        () async {
      final samples =
          getIrisDataFrame().shuffle().dropSeries(names: ['Id']);
      final pipeline = Pipeline(samples, [
        toIntegerLabels(
          columnNames: ['Species'],
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

    test(
        'should save graphical representation as svg image, pima indians diabetes dataset',
        () async {
      final samples = getPimaIndiansDiabetesDataFrame().shuffle();
      final classifier = DecisionTreeClassifier(
        samples,
        'Outcome',
        minError: 0.15,
        minSamplesCount: 1,
        maxDepth: 5,
      );

      await classifier
          .saveAsSvg('e2e/decision_tree_classifier/pima_indians_tree.svg');
    });
  });
}
