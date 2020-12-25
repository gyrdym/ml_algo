import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

abstract class SoftmaxRegressorFactory {
  SoftmaxRegressor create({
    DataFrame trainData,
    List<String> targetNames,
    LinearOptimizerType optimizerType,
    int iterationsLimit,
    double initialLearningRate,
    double minCoefficientsUpdate,
    double lambda,
    RegularizationType regularizationType,
    int randomSeed,
    int batchSize,
    bool fitIntercept,
    double interceptScale,
    LearningRateType learningRateType,
    bool isFittingDataNormalized,
    InitialCoefficientsType initialCoefficientsType,
    Matrix initialCoefficients,
    num positiveLabel,
    num negativeLabel,
    bool collectLearningData,
    DType dtype,
  });

  SoftmaxRegressor fromJson(String json);
}
