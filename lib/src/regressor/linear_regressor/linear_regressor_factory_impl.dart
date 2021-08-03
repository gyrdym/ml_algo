import 'dart:convert';

import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/regressor/_helpers/squared_cost_optimizer_factory.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_factory.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_impl.dart';
import 'package:ml_algo/src/regressor/linear_regressor/migrations/migrate_linear_regressor_schema_v1_to_v2.dart';
import 'package:ml_dataframe/src/data_frame/data_frame.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

class LinearRegressorFactoryImpl implements LinearRegressorFactory {
  const LinearRegressorFactoryImpl();

  @override
  LinearRegressor create({
    required DataFrame fittingData,
    required String targetName,
    LinearOptimizerType optimizerType = LinearOptimizerType.gradient,
    int iterationsLimit = 100,
    LearningRateType learningRateType = LearningRateType.constant,
    InitialCoefficientsType initialCoefficientsType =
        InitialCoefficientsType.zeroes,
    double initialLearningRate = 1e-3,
    double minCoefficientsUpdate = 1e-12,
    double lambda = 0,
    bool fitIntercept = false,
    double interceptScale = 1.0,
    int batchSize = 1,
    bool isFittingDataNormalized = false,
    bool collectLearningData = false,
    DType dtype = DType.float32,
    RegularizationType? regularizationType,
    int? randomSeed,
    Matrix? initialCoefficients,
  }) {
    final optimizer = createSquaredCostOptimizer(
      fittingData,
      targetName,
      optimizerType: optimizerType,
      iterationsLimit: iterationsLimit,
      initialLearningRate: initialLearningRate,
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
      dtype: dtype,
    );

    final coefficients = optimizer
        .findExtrema(
          initialCoefficients: initialCoefficients,
          isMinimizingObjective: true,
          collectLearningData: collectLearningData,
        )
        .getColumn(0);
    final costPerIteration = optimizer.costPerIteration;
    final targetIndex = [...fittingData.header].indexOf(targetName);

    return LinearRegressorImpl(
      coefficients,
      targetName,
      targetIndex,
      optimizerType: optimizerType,
      iterationsLimit: iterationsLimit,
      learningRateType: learningRateType,
      initialCoefficientsType: initialCoefficientsType,
      initialLearningRate: initialLearningRate,
      minCoefficientsUpdate: minCoefficientsUpdate,
      lambda: lambda,
      regularizationType: regularizationType,
      randomSeed: randomSeed,
      batchSize: batchSize,
      initialCoefficients: initialCoefficients,
      isFittingDataNormalized: isFittingDataNormalized,
      fitIntercept: fitIntercept,
      interceptScale: interceptScale,
      costPerIteration: costPerIteration,
      dtype: dtype,
    );
  }

  @override
  LinearRegressor fromJson(String json) {
    final v1Schema = jsonDecode(json) as Map<String, dynamic>;
    final v2Schema = migrateLinearRegressorSchemaV1toV2(v1Schema);

    return LinearRegressorImpl.fromJson(v2Schema);
  }
}
