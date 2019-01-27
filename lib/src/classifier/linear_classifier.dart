import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/multinomial_type.dart';
import 'package:ml_algo/predictor.dart';
import 'package:ml_algo/src/classifier/logistic_regressor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

abstract class LinearClassifier<T> implements Predictor<T> {
  factory LinearClassifier.logisticRegressor({
    int iterationLimit,
    double learningRate,
    double minWeightsUpdate,
    double lambda,
    int batchSize,
    int randomSeed,
    bool fitIntercept,
    double interceptScale,
    MultinomialType multinomialType,
    LearningRateType learningRateType}) = LogisticRegressor<T>;

  factory LinearClassifier.SVM() => throw UnimplementedError();
  factory LinearClassifier.NaiveBayes() => throw UnimplementedError();

  MLMatrix<T> predictProbabilities(MLMatrix<T> features);
  MLVector<T> predictClasses(MLMatrix<T> features);
}