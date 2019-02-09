import 'package:ml_algo/gradient_type.dart';
import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/predictor.dart';
import 'package:ml_algo/src/classifier/logistic_regressor.dart';
import 'package:ml_algo/src/optimizer/optimizer_type.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

abstract class LinearClassifier implements Predictor {
  factory LinearClassifier.logisticRegressor({
    int iterationLimit,
    double learningRate,
    double minWeightsUpdate,
    double lambda,
    int randomSeed,
    int batchSize,
    bool fitIntercept,
    double interceptScale,
    LearningRateType learningRateType,
    OptimizerType optimizer,
    GradientType gradientType,
    Type dtype,
  }) = LogisticRegressor;

  factory LinearClassifier.SVM() => throw UnimplementedError();
  factory LinearClassifier.NaiveBayes() => throw UnimplementedError();

  MLMatrix predictProbabilities(MLMatrix features);
  MLVector predictClasses(MLMatrix features);
}