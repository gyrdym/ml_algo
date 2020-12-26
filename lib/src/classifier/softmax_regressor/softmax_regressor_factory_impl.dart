import 'dart:convert';

import 'package:ml_algo/src/classifier/_helpers/create_log_likelihood_optimizer.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_impl.dart';
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

class SoftmaxRegressorFactoryImpl implements SoftmaxRegressorFactory {
  const SoftmaxRegressorFactoryImpl();

  @override
  SoftmaxRegressor create({
    DataFrame trainData,
    Iterable<String> targetNames,
    LinearOptimizerType optimizerType = LinearOptimizerType.gradient,
    int iterationsLimit = 100,
    double initialLearningRate = 1e-3,
    double minCoefficientsUpdate = 1e-12,
    double lambda,
    RegularizationType regularizationType,
    int randomSeed,
    int batchSize = 1,
    bool fitIntercept = false,
    double interceptScale = 1.0,
    LearningRateType learningRateType = LearningRateType.constant,
    bool isFittingDataNormalized,
    InitialCoefficientsType initialCoefficientsType =
        InitialCoefficientsType.zeroes,
    Matrix initialCoefficients,
    num positiveLabel = 1,
    num negativeLabel = 0,
    bool collectLearningData = false,
    DType dtype = DType.float32,
  }) {
    if (targetNames.isNotEmpty && targetNames.length < 2) {
      throw Exception('The target column should be encoded properly '
          '(e.g., via one-hot encoder)');
    }

    validateTrainData(trainData, targetNames);

    final linkFunction = softmaxRegressorInjector
        .get<LinkFunction>(
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
      positiveLabel: positiveLabel,
      negativeLabel: negativeLabel,
      dtype: dtype,
    );
    final coefficientsByClasses = optimizer.findExtrema(
      initialCoefficients: initialCoefficients,
      isMinimizingObjective: false,
      collectLearningData: collectLearningData,
    );
    final costPerIteration = optimizer
        .costPerIteration
        .isNotEmpty
        ? optimizer.costPerIteration
        : null;

    return SoftmaxRegressorImpl(
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
      coefficientsByClasses,
      targetNames,
      linkFunction,
      fitIntercept,
      interceptScale,
      positiveLabel,
      negativeLabel,
      costPerIteration,
      dtype,
    );
  }

  @override
  SoftmaxRegressor fromJson(String json) {
    final decoded = jsonDecode(json) as Map<String, dynamic>;

    return SoftmaxRegressorImpl.fromJson(decoded);
  }
}
