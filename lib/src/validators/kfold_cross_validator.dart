import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/predictors/predictor_interface.dart';
import 'package:dart_ml/src/estimators/estimator_interface.dart';
import 'package:dart_ml/src/data_splitters/k_fold_splitter.dart';

class KFoldCrossValidator {
  final KFoldSplitter _splitter = new KFoldSplitter();

  VectorInterface validate(PredictorInterface predictor, List<VectorInterface> features, VectorInterface labels,
                           {EstimatorInterface estimator, int numberOfFolds = 5}) {

    if (features.length != labels.length) {
      throw new Exception('Number of features objects must be equal to the number of labels!');
    }

    List<List<int>> testSampleRanges = _splitter.split(features.length, numberOfFolds: numberOfFolds);
    List<double> scores = [];

    for (int i = 0; i < testSampleRanges.length; i++) {
      List<int> testRange = testSampleRanges[i];

      List<VectorInterface> trainFeatures = features.sublist(0, testRange.first)
        ..addAll(features.sublist(testRange.last));
      VectorInterface trainLabels = labels.cut(0, testRange.first)
        ..concat(labels.cut(testRange.last));

      List<VectorInterface> testFeatures = features.sublist(testRange.first, testRange.last);
      VectorInterface testLabels = labels.cut(testRange.first, testRange.last);

      predictor.train(trainFeatures, trainLabels, trainFeatures.first.copy()..fill(0.0));
      scores.add(predictor.test(testFeatures, testLabels, estimator: estimator));
    }

    return labels.createFrom(scores);
  }
}
