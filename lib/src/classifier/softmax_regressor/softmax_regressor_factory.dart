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
    required DataFrame trainData,
    required Iterable<String> targetNames,
    required LinearOptimizerType optimizerType,
    required int iterationsLimit,
    required double initialLearningRate,
    required double minCoefficientsUpdate,
    required double lambda,
    required int batchSize,
    required bool fitIntercept,
    required double interceptScale,
    required LearningRateType learningRateType,
    required bool isFittingDataNormalized,
    required InitialCoefficientsType initialCoefficientsType,
    required num positiveLabel,
    required num negativeLabel,
    required bool collectLearningData,
    required DType dtype,
    RegularizationType? regularizationType,
    Matrix? initialCoefficients,
    int? randomSeed,
  });

  SoftmaxRegressor fromJson(String json);
}
