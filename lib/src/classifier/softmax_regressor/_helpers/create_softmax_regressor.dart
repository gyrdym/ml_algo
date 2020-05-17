import 'package:ml_algo/src/classifier/_helpers/log_likelihood_optimizer_factory.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory.dart';
import 'package:ml_algo/src/di/dependencies.dart';
import 'package:ml_algo/src/helpers/validate_train_data.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_dependency_tokens.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

SoftmaxRegressor createSoftmaxRegressor(
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
    DType dtype,
) {
  if (targetNames.isNotEmpty && targetNames.length < 2) {
    throw Exception('The target column should be encoded properly '
        '(e.g., via one-hot encoder)');
  }

  validateTrainData(trainData, targetNames);

  final linkFunction = dependencies.getDependency<LinkFunction>(
      dependencyName: dTypeToSoftmaxLinkFunctionToken[dtype]);

  final optimizer = createLogLikelihoodOptimizer(
    trainData,
    targetNames,
    linkFunction,
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
    initialCoefficients: initialCoefficients,
    isMinimizingObjective: false,
  );

  final regressorFactory = dependencies
      .getDependency<SoftmaxRegressorFactory>();

  return regressorFactory.create(
    coefficientsByClasses,
    targetNames,
    linkFunction,
    fitIntercept,
    interceptScale,
    positiveLabel,
    negativeLabel,
    dtype,
  );
}
