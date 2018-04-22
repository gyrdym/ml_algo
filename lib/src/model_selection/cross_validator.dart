import 'package:dart_ml/src/data_splitter/factory.dart';
import 'package:dart_ml/src/data_splitter/splitter.dart';
import 'package:dart_ml/src/data_splitter/type.dart';
import 'package:dart_ml/src/metric/type.dart';
import 'package:dart_ml/src/model_selection/evaluable.dart';
import 'package:simd_vector/vector.dart' show Vector;

class CrossValidator<T extends Vector> {
  final Splitter _splitter;

  factory CrossValidator.KFold({int numberOfFolds = 5}) =>
      new CrossValidator._(SplitterType.KFOLD, numberOfFolds);

  factory CrossValidator.LPO({int p = 5}) =>
      new CrossValidator._(SplitterType.LPO, p);

  CrossValidator._(SplitterType splitterType, int value) :
        _splitter = DataSplitterFactory.createByType(splitterType, value);

  double evaluate(
    Evaluable predictor,
    List<T> features,
    List<double> labels,
    MetricType metric,
    {bool isDataNormalized = false}
  ) {

    if (features.length != labels.length) {
      throw new Exception('Number of features objects must be equal to the number of labels!');
    }

    final allIndicesGroups = _splitter.split(features.length);
    final scores = new List<double>(allIndicesGroups.length);
    int scoreCounter = 0;

    for (final testIndices in allIndicesGroups) {
      final trainFeatures = new List<T>(features.length - testIndices.length);
      final trainLabels = new List<double>.filled(features.length - testIndices.length, 0.0);
      final testFeatures = new List<T>(testIndices.length);
      final testLabels = new List<double>.filled(testIndices.length, 0.0);
      int trainSamplesCounter = 0;
      int testSamplesCounter = 0;

      for (int index = 0; index < features.length; index++) {
        if (testIndices.contains(index)) {
          testFeatures[testSamplesCounter] = features[index];
          testLabels[testSamplesCounter] = labels[index];
          testSamplesCounter++;
        } else {
          trainFeatures[trainSamplesCounter] = features[index];
          trainLabels[trainSamplesCounter] = labels[index];
          trainSamplesCounter++;
        }
      }

      predictor.fit(trainFeatures, trainLabels, isDataNormalized: isDataNormalized);

      scores[scoreCounter++] = predictor.test(testFeatures, testLabels, metric);
    }

    return scores.reduce((sum, value) => (sum ?? 0.0) + value) / scores.length;
  }
}
