import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

LinearOptimizer createLogLikelihoodOptimizer(
    DataFrame fittingData,
    Iterable<String> targetNames,
    LinkFunctionType linkFunctionType, {
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
  DType dtype,
}) {
  final splits = featuresTargetSplit(fittingData,
    targetNames: targetNames,
  ).toList();

  final points = splits[0].toMatrix();
  final labels = splits[1].toMatrix();

  final optimizerFactory = dependencies
      .getDependency<LinearOptimizerFactory>();

  final linkFunctionFactory = dependencies
      .getDependency<LinkFunctionFactory>();

  final linkFunction = linkFunctionFactory
      .createByType(linkFunctionType, dtype: dtype);

  final costFunctionFactory = dependencies
      .getDependency<CostFunctionFactory>();

  final costFunction = costFunctionFactory.createByType(
    CostFunctionType.logLikelihood,
    linkFunction: linkFunction,
  );

  return optimizerFactory.createByType(
    optimizerType,
    addInterceptIf(fitIntercept, points, interceptScale),
    labels,
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
