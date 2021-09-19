import 'dart:convert';

import 'package:ml_algo/src/classifier/_helpers/create_log_likelihood_optimizer.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/migrations/migrate_softmax_regressor_schema_v2_to_v3.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/migrations/migrate_softmax_regressor_schema_v3_to_v4.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_impl.dart';
import 'package:ml_algo/src/common/constants/default_parameters/classification.dart';
import 'package:ml_algo/src/common/constants/default_parameters/common.dart';
import 'package:ml_algo/src/common/constants/default_parameters/coordinate_optimization.dart';
import 'package:ml_algo/src/common/constants/default_parameters/gradient_optimization.dart';
import 'package:ml_algo/src/common/constants/default_parameters/linear_optimization.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

class SoftmaxRegressorFactoryImpl implements SoftmaxRegressorFactory {
  const SoftmaxRegressorFactoryImpl(this._linkFunction);

  final LinkFunction _linkFunction;

  @override
  SoftmaxRegressor create({
    required DataFrame trainData,
    required Iterable<String> targetNames,
    LinearOptimizerType optimizerType = linearOptimizerTypeDefaultValue,
    int iterationsLimit = iterationLimitDefaultValue,
    double initialLearningRate = initialLearningRateDefaultValue,
    double decay = decayDefaultValue,
    int dropRate = dropRateDefaultValue,
    double minCoefficientsUpdate = minCoefficientsUpdateDefaultValue,
    double lambda = lambdaDefaultValue,
    RegularizationType? regularizationType,
    int? randomSeed,
    int batchSize = batchSizeDefaultValue,
    bool fitIntercept = fitInterceptDefaultValue,
    double interceptScale = interceptScaleDefaultValue,
    LearningRateType learningRateType = learningRateTypeDefaultValue,
    bool isFittingDataNormalized = isFittingDataNormalizedDefaultValue,
    InitialCoefficientsType initialCoefficientsType =
        initialCoefficientsTypeDefaultValue,
    Matrix? initialCoefficients,
    num positiveLabel = positiveLabelDefaultValue,
    num negativeLabel = negativeLabelDefaultValue,
    bool collectLearningData = collectLearningDataDefaultValue,
    DType dtype = dTypeDefaultValue,
  }) {
    if (targetNames.isNotEmpty && targetNames.length < 2) {
      throw Exception('The target column should be encoded properly '
          '(e.g., via one-hot encoder)');
    }

    final optimizer = createLogLikelihoodOptimizer(
      trainData,
      targetNames,
      _linkFunction,
      optimizerType: optimizerType,
      iterationsLimit: iterationsLimit,
      initialLearningRate: initialLearningRate,
      decay: decay,
      dropRate: dropRate,
      minCoefficientsUpdate: minCoefficientsUpdate,
      lambda: lambda,
      regularizationType: regularizationType,
      randomSeed: randomSeed,
      batchSize: batchSize,
      learningRateType: learningRateType,
      initialCoefficientsType: initialCoefficientsType,
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
    final costPerIteration = optimizer.costPerIteration.isNotEmpty
        ? optimizer.costPerIteration
        : null;

    return SoftmaxRegressorImpl(
      optimizerType,
      iterationsLimit,
      initialLearningRate,
      decay,
      dropRate,
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
      _linkFunction,
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
    final v2Schema = jsonDecode(json) as Map<String, dynamic>;
    final v3Schema = migrateSoftmaxRegressorSchemaV2toV3(v2Schema);
    final v4Schema = migrateSoftmaxRegressorSchemaV3toV4(v3Schema);

    return SoftmaxRegressorImpl.fromJson(v4Schema);
  }
}
