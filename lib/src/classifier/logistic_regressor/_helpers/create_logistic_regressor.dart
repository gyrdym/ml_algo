import 'package:ml_algo/src/classifier/_helpers/log_likelihood_optimizer_factory.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_factory.dart';
import 'package:ml_algo/src/di/dependencies.dart';
import 'package:ml_algo/src/helpers/validate_initial_coefficients.dart';
import 'package:ml_algo/src/helpers/validate_train_data.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

LogisticRegressor createLogisticRegressor(
    DataFrame trainData,
    String targetName,
    LinearOptimizerType optimizerType,
    int iterationsLimit,
    double initialLearningRate,
    double minCoefficientsUpdate,
    double probabilityThreshold,
    double lambda,
    RegularizationType regularizationType,
    int randomSeed,
    int batchSize,
    bool fitIntercept,
    double interceptScale,
    bool isFittingDataNormalized,
    LearningRateType learningRateType,
    InitialCoefficientsType initialCoefficientsType,
    Vector initialCoefficients,
    num positiveLabel,
    num negativeLabel,
    DType dtype,
) {
  validateTrainData(trainData, [targetName]);

  if (initialCoefficients.isNotEmpty) {
    validateInitialCoefficients(initialCoefficients, fitIntercept,
        trainData.toMatrix(dtype).columnsNum - 1);
  }

  final linkFunctionFactory = dependencies
      .getDependency<LinkFunctionFactory>();

  final linkFunction = linkFunctionFactory
      .createByType(LinkFunctionType.inverseLogit, dtype: dtype);

  final optimizer = createLogLikelihoodOptimizer(
    trainData,
    [targetName],
    LinkFunctionType.inverseLogit,
    optimizerType: optimizerType,
    iterationsLimit: iterationsLimit,
    initialLearningRate: initialLearningRate,
    minCoefficientsUpdate: minCoefficientsUpdate,
    lambda: lambda,
    regularizationType: regularizationType,
    randomSeed: randomSeed,
    batchSize: batchSize,
    learningRateType: learningRateType,
    initialWeightsType: initialCoefficientsType,
    fitIntercept: fitIntercept,
    interceptScale: interceptScale,
    isFittingDataNormalized: isFittingDataNormalized,
    dtype: dtype,
  );

  final coefficientsByClasses = optimizer.findExtrema(
    initialCoefficients: initialCoefficients.isNotEmpty
        ? Matrix.fromColumns([initialCoefficients], dtype: dtype)
        : null,
    isMinimizingObjective: false,
  );

  final regressorFactory = dependencies
      .getDependency<LogisticRegressorFactory>();

  return regressorFactory.create(
    targetName,
    linkFunction,
    probabilityThreshold,
    fitIntercept,
    interceptScale,
    coefficientsByClasses,
    negativeLabel,
    positiveLabel,
    dtype,
  );
}
