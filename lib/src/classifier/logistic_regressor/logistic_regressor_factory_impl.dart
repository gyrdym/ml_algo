import 'dart:convert';

import 'package:ml_algo/src/classifier/_helpers/create_log_likelihood_optimizer.dart';
import 'package:ml_algo/src/classifier/_helpers/validate_linear_classification_optimizer_type.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_factory.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_impl.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/migrations/migrate_logistic_regressor_schema_v2_to_v3.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/migrations/migrate_logistic_regressor_schema_v3_to_v4.dart';
import 'package:ml_algo/src/common/constants/default_parameters/classification.dart';
import 'package:ml_algo/src/common/constants/default_parameters/common.dart';
import 'package:ml_algo/src/common/constants/default_parameters/coordinate_optimization.dart';
import 'package:ml_algo/src/common/constants/default_parameters/gradient_optimization.dart';
import 'package:ml_algo/src/common/constants/default_parameters/linear_optimization.dart';
import 'package:ml_algo/src/helpers/validate_class_labels.dart';
import 'package:ml_algo/src/helpers/validate_initial_coefficients.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_dataframe/src/data_frame/data_frame.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class LogisticRegressorFactoryImpl implements LogisticRegressorFactory {
  const LogisticRegressorFactoryImpl(this._linkFunction);

  final LinkFunction _linkFunction;

  @override
  LogisticRegressor create({
    required DataFrame trainData,
    required String targetName,
    LinearOptimizerType optimizerType = linearOptimizerTypeDefaultValue,
    int iterationsLimit = iterationLimitDefaultValue,
    double initialLearningRate = initialLearningRateDefaultValue,
    int dropRate = dropRateDefaultValue,
    double decay = decayDefaultValue,
    double minCoefficientsUpdate = minCoefficientsUpdateDefaultValue,
    double probabilityThreshold = probabilityThresholdDefaultValue,
    double lambda = lambdaDefaultValue,
    int batchSize = batchSizeDefaultValue,
    bool fitIntercept = fitInterceptDefaultValue,
    double interceptScale = interceptScaleDefaultValue,
    bool isFittingDataNormalized = isFittingDataNormalizedDefaultValue,
    LearningRateType learningRateType = learningRateTypeDefaultValue,
    InitialCoefficientsType initialCoefficientsType =
        initialCoefficientsTypeDefaultValue,
    num positiveLabel = positiveLabelDefaultValue,
    num negativeLabel = negativeLabelDefaultValue,
    bool collectLearningData = collectLearningDataDefaultValue,
    DType dtype = dTypeDefaultValue,
    RegularizationType? regularizationType,
    Vector? initialCoefficients,
    int? randomSeed,
  }) {
    validateLinearClassificationOptimizerType(optimizerType);
    validateClassLabels(positiveLabel, negativeLabel);

    if (initialCoefficients?.isNotEmpty == true) {
      validateInitialCoefficients(initialCoefficients!, fitIntercept,
          trainData.toMatrix(dtype).columnCount - 1);
    }

    final optimizer = createLogLikelihoodOptimizer(
      trainData,
      [targetName],
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
      initialCoefficients: initialCoefficients?.isNotEmpty == true
          ? Matrix.fromColumns([initialCoefficients!], dtype: dtype)
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
      [targetName],
      _linkFunction,
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

  @override
  LogisticRegressor fromJson(String json) {
    final v2Schema = jsonDecode(json) as Map<String, dynamic>;
    final v3Schema = migrateLogisticRegressorSchemaV2toV3(v2Schema);
    final v4Schema = migrateLogisticRegressorSchemaV3toV4(v3Schema);

    return LogisticRegressorImpl.fromJson(v4Schema);
  }
}
