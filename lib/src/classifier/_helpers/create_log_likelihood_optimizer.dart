import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/di/injector.dart';
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
      required LinearOptimizerType optimizerType,
      required int iterationsLimit,
      required double initialLearningRate,
      required double minCoefficientsUpdate,
//      required double probabilityThreshold,
      required double lambda,
      RegularizationType? regularizationType,
      int? randomSeed,
      required int batchSize,
      required bool fitIntercept,
      required double interceptScale,
      required bool isFittingDataNormalized,
      required LearningRateType learningRateType,
      required InitialCoefficientsType initialCoefficientsType,
//      required Matrix initialWeights,
      required num positiveLabel,
      required num negativeLabel,
      required DType dtype,
}) {
  validateClassLabels(positiveLabel, negativeLabel);

  final splits = featuresTargetSplit(fittingData, targetNames: targetNames)
      .toList();
  final points = splits[0].toMatrix(dtype);
  final labels = splits[1].toMatrix(dtype);
  final optimizerFactory = injector.get<LinearOptimizerFactory>();
  final costFunctionFactory = injector.get<CostFunctionFactory>();
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
    initialCoefficientsType: initialCoefficientsType,
    dtype: dtype,
    isFittingDataNormalized: isFittingDataNormalized,
  );
}
