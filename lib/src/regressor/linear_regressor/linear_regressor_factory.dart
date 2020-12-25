import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

abstract class LinearRegressorFactory {
  LinearRegressor create({
    DataFrame fittingData,
    String targetName,
    LinearOptimizerType optimizerType,
    int iterationsLimit,
    LearningRateType learningRateType,
    InitialCoefficientsType initialCoefficientsType,
    double initialLearningRate,
    double minCoefficientsUpdate,
    double lambda,
    RegularizationType regularizationType,
    bool fitIntercept,
    double interceptScale,
    int randomSeed,
    int batchSize,
    Matrix initialCoefficients,
    bool isFittingDataNormalized,
    bool collectLearningData,
    DType dtype,
  });

  LinearRegressor fromJson(String json);
}
