import 'package:dart_ml/src/core/interface.dart' show Splitter, SplitterType, MetricType, Predictor;
import 'package:dart_ml/src/core/implementation.dart';
import 'package:dart_ml/src/di/injector.dart';
import 'package:simd_vector/vector.dart' show Float32x4Vector;
import 'package:di/di.dart';

class CrossValidator {
  Splitter _splitter;

  factory CrossValidator.KFold({int numberOfFolds = 5}) =>
      new CrossValidator._(SplitterType.KFOLD, numberOfFolds);

  factory CrossValidator.LPO({int p = 5}) =>
      new CrossValidator._(SplitterType.LPO, p);

  CrossValidator._(SplitterType splitterType, int value) {
    modelSelectionInjector ??= new ModuleInjector(<Module>[ModuleFactory
        .modelSelectionModule(value, splitter: splitterType)]);

    _splitter = modelSelectionInjector.get(Splitter);
  }

  Float32x4Vector evaluate(Predictor predictor, List<Float32x4Vector> features, List<double> labels,
                           {MetricType metric}) {

    if (features.length != labels.length) {
      throw new Exception('Number of features objects must be equal to the number of labels!');
    }

    Iterable<Iterable<int>> allIndices = _splitter.split(features.length);
    List<double> scores = new List<double>(allIndices.length);
    int scoreCounter = 0;

    for (Iterable<int> testIndices in allIndices) {
      List<Float32x4Vector> trainFeatures = new List<Float32x4Vector>(features.length - testIndices.length);
      List<double> trainLabels = new List<double>.filled(features.length - testIndices.length, 0.0);

      List<Float32x4Vector> testFeatures = new List<Float32x4Vector>(testIndices.length);
      List<double> testLabels = new List<double>.filled(testIndices.length, 0.0);

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

      predictor.train(trainFeatures, trainLabels);

      scores[scoreCounter++] = predictor.test(testFeatures, testLabels, metric: metric);
    }

    return new Float32x4Vector.from(scores);
  }
}
