import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/di/dependencies.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_algo/src/helpers/normalize_class_labels.dart';
import 'package:ml_algo/src/helpers/validate_class_labels.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

LinearOptimizer createLogLikelihoodOptimizer(
    DataFrame fittingData,
    Iterable<String> targetNames,
    LinkFunction linkFunction, {
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
  InitialCoefficientsType initialWeightsType,
  Matrix initialWeights,
  num positiveLabel,
  num negativeLabel,
  DType dtype,
}) {
  validateClassLabels(positiveLabel, negativeLabel);

  final splits = featuresTargetSplit(fittingData, targetNames: targetNames)
      .toList();
  final points = splits[0].toMatrix(dtype);
  final labels = splits[1].toMatrix(dtype);
  final optimizerFactory = dependencies.get<LinearOptimizerFactory>();
  final costFunctionFactory = dependencies.get<CostFunctionFactory>();
  final costFunction = costFunctionFactory.createByType(
    CostFunctionType.logLikelihood,
    linkFunction: linkFunction,
    positiveLabel: positiveLabel,
    negativeLabel: negativeLabel,
  );
  final normalizedLabels = normalizeClassLabels(labels,
      positiveLabel, negativeLabel);

  return optimizerFactory.createByType(
    optimizerType,
    addInterceptIf(fitIntercept, points, interceptScale, dtype),
    normalizedLabels,
    costFunction: costFunction,
    iterationLimit: iterationsLimit,
    initialLearningRate: initialLearningRate,
    minCoefficientsUpdate: minCoefficientsUpdate,
    lambda: lambda,
    regularizationType: regularizationType,
    randomSeed: randomSeed,
    batchSize: batchSize,
    learningRateType: learningRateType,
    initialCoefficientsType: initialWeightsType,
    dtype: dtype,
    isFittingDataNormalized: isFittingDataNormalized,
  );
}
