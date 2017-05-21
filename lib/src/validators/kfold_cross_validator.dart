import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/predictors/predictor_interface.dart';
import 'package:dart_ml/src/estimators/estimator_interface.dart';
import 'package:dart_ml/src/data_splitters/k_fold_splitter.dart';

class KFoldCrossValidator {
  final KFoldSplitter _splitter;

  KFoldCrossValidator({int numberOfFolds = 5}) : _splitter = new KFoldSplitter(numberOfFolds: numberOfFolds);

  VectorInterface validate(PredictorInterface predictor, List<VectorInterface> features, VectorInterface labels,
                           {EstimatorInterface estimator}) {

    if (features.length != labels.length) {
      throw new Exception('Number of features objects must be equal to the number of labels!');
    }

    Iterable<Iterable<int>> allIndices = _splitter.split(features.length);
    List<double> scores = new List<double>(allIndices.length);
    int scoreCounter = 0;

    for (var testIndices in allIndices) {
      List<VectorInterface> trainFeatures = new List<VectorInterface>(features.length - testIndices.length);
      VectorInterface trainLabels = labels.copy()
        ..length = features.length - testIndices.length
        ..fill(0.0);

      List<VectorInterface> testFeatures = new List<VectorInterface>(testIndices.length);
      VectorInterface testLabels = labels.copy()
        ..length = testIndices.length
        ..fill(0.0);

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

      predictor.train(trainFeatures, trainLabels, trainFeatures.first.copy()
        ..fill(0.0));

      scores[scoreCounter++] = predictor.test(testFeatures, testLabels, estimator: estimator);
    }

    return labels.createFrom(scores);
  }
}
