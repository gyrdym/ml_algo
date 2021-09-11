import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

abstract class LinearRegressorFactory {
  LinearRegressor create({
    required DataFrame fittingData,
    required String targetName,
    required LinearOptimizerType optimizerType,
    required int iterationsLimit,
    required LearningRateType learningRateType,
    required InitialCoefficientsType initialCoefficientsType,
    required double initialLearningRate,
    required double decay,
    required double minCoefficientsUpdate,
    required double lambda,
    required bool fitIntercept,
    required double interceptScale,
    required int batchSize,
    required bool isFittingDataNormalized,
    required bool collectLearningData,
    required DType dtype,
    RegularizationType? regularizationType,
    int? randomSeed,
    Matrix? initialCoefficients,
  });

  LinearRegressor fromJson(String json);
}
