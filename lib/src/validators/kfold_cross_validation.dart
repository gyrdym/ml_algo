import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/predictors/predictor_interface.dart';
import 'package:dart_ml/src/validators/cross_validator_interface.dart';
import 'package:dart_ml/src/data_splitters/k_fold_splitter.dart';

class KFoldCrossValidator implements CrossValidatorInterface {
  final KFoldSplitter _splitter = new KFoldSplitter();

  List<double> validate(PredictorInterface predictor, List<VectorInterface> features, List<double> labels,
                        {int numberOfFolds = 5}) {

    if (features.length != labels.length) {
      throw new Exception('Number of features objects must be equal to the number of labels!');
    }

    List<List<int>> testSampleRanges = _splitter.split(features.length);
    List<double> scores = [];

    for (int i = 0; i < testSampleRanges.length; i++) {
      List<int> testRange = testSampleRanges[i];

      List<VectorInterface> trainFeatures = features.sublist(0, testRange.first)
        ..addAll(features.sublist(testRange.last));
      List<double> trainLabels = labels.sublist(0, testRange.first)
        ..addAll(labels.sublist(testRange.last));

      List<VectorInterface> testFeatures = features.sublist(testRange.first, testRange.last);
      List<double> testLabels = labels.sublist(testRange.first, testRange.last);

      predictor.train(trainFeatures, trainLabels, trainFeatures.first.copy()..fill(0.0));
      predictor.predict(testFeatures, testFeatures.first.copy()..fill(0.0));
    }

    return scores;
  }
}
