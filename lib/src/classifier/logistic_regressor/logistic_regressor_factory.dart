import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';

abstract class LogisticRegressorFactory {
  LogisticRegressor create({
    required DataFrame trainData,
    required String targetName,
    required LinearOptimizerType optimizerType,
    required int iterationsLimit,
    required double initialLearningRate,
    required double decay,
    required int dropRate,
    required double minCoefficientsUpdate,
    required double probabilityThreshold,
    required double lambda,
    required int batchSize,
    required bool fitIntercept,
    required double interceptScale,
    required bool isFittingDataNormalized,
    required LearningRateType learningRateType,
    required InitialCoefficientsType initialCoefficientsType,
    required num positiveLabel,
    required num negativeLabel,
    required bool collectLearningData,
    required DType dtype,
    RegularizationType? regularizationType,
    Vector? initialCoefficients,
    int? randomSeed,
  });

  LogisticRegressor fromJson(String json);
}
