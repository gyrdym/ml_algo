import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:test/test.dart';

void main() async {
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

  group('DecisionTreeClassifier', () {
    test('should save graphical representation as svg image', () async {
      await classifier.saveAsSvg('e2e/decision_tree_classifier/tree.svg');
    });
  });
}
