import 'package:ml_algo/src/classifier/_helpers/create_log_likelihood_optimizer.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_impl.dart';
import 'package:ml_algo/src/helpers/validate_class_labels.dart';
import 'package:ml_algo/src/helpers/validate_initial_coefficients.dart';
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
import 'package:ml_linalg/vector.dart';

LogisticRegressor createLogisticRegressor({
  DataFrame trainData,
  String targetName,
  LinearOptimizerType optimizerType = LinearOptimizerType.gradient,
  int iterationsLimit = 100,
  double initialLearningRate = 1e-3,
  double minCoefficientsUpdate = 1e-12,
  double probabilityThreshold = 0.5,
  double lambda = 0.0,
  RegularizationType regularizationType,
  int randomSeed,
  int batchSize = 1,
  bool fitIntercept = false,
  double interceptScale = 1.0,
  bool isFittingDataNormalized = false,
  LearningRateType learningRateType = LearningRateType.constant,
  InitialCoefficientsType initialCoefficientsType =
      InitialCoefficientsType.zeroes,
  Vector initialCoefficients,
  num positiveLabel = 1,
  num negativeLabel = 0,
  bool collectLearningData = false,
  DType dtype = DType.float32,
}) {
  validateTrainData(trainData, [targetName]);
  validateClassLabels(positiveLabel, negativeLabel);

  if (initialCoefficients.isNotEmpty) {
    validateInitialCoefficients(initialCoefficients, fitIntercept,
        trainData.toMatrix(dtype).columnsNum - 1);
  }

  final linkFunction = logisticRegressorInjector.get<LinkFunction>(
      dependencyName: dTypeToInverseLogitLinkFunctionToken[dtype]);
  final optimizer = createLogLikelihoodOptimizer(
    trainData,
    [targetName],
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
    positiveLabel: positiveLabel,
    negativeLabel: negativeLabel,
    dtype: dtype,
  );
  final coefficientsByClasses = optimizer.findExtrema(
    initialCoefficients: initialCoefficients.isNotEmpty
        ? Matrix.fromColumns([initialCoefficients], dtype: dtype)
        : null,
    isMinimizingObjective: false,
    collectLearningData: collectLearningData,
  );
  final costPerIteration = optimizer.costPerIteration.isNotEmpty
      ? optimizer.costPerIteration
      : null;

  return LogisticRegressorImpl(
    optimizerType,
    iterationsLimit,
    initialLearningRate,
    minCoefficientsUpdate,
    lambda,
    regularizationType,
    randomSeed,
    batchSize,
    isFittingDataNormalized,
    learningRateType,
    initialCoefficientsType,
    initialCoefficients,
    [targetName],
    linkFunction,
    fitIntercept,
    interceptScale,
    coefficientsByClasses,
    probabilityThreshold,
    negativeLabel,
    positiveLabel,
    costPerIteration,
    dtype,
  );
}
