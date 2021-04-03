import 'dart:io';

import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type_json_encoded_values.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type_json_encoded_values.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type_json_encoded_values.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type_json_encoded_values.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_constants.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_json_keys.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/dtype_to_json.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

void main() {
  group('LinearRegressor', () {
    final featureNames = ['feature 1', 'feature 2', 'feature 3'];
    final optimizerType = LinearOptimizerType.gradient;
    final learningRateType = LearningRateType.decreasingAdaptive;
    final initialCoefficientsType = InitialCoefficientsType.zeroes;
    final initialLearningRate = 9.55;
    final minCoefficientsUpdate = 1.5;
    final lambda = 133.5;
    final regularizationType = RegularizationType.L2;
    final randomSeed = 4001;
    final batchSize = 2;
    final initialCoefficients = Matrix.column([1, 1, 1, 1], dtype: DType.float64);
    final isFittingDataNormalized = true;
    final targetName = 'outcome';
    final dataSource = <Iterable>[
      <String>[...featureNames, targetName],
      <num>[ 100.5,    45,     1, -1000.08],
      <num>[  4301, -1708, 10001,        1],
      <num>[-100.3,    -1, 10597,    10003],
      <num>[  3003,     0,  1204,        0],
    ];
    final dataSet = DataFrame(dataSource, headerExists: true);
    final fitIntercept = true;
    final interceptScale = 10032.0;
    final iterationsLimit = 2;
    final dtype = DType.float64;
    final filePath = 'test/regressor/linear_regressor.json';

    late LinearRegressor regressor;

    setUp(() {
      regressor = LinearRegressor(dataSet, targetName,
        optimizerType: optimizerType,
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
        iterationsLimit: iterationsLimit,
        collectLearningData: true,
        dtype: dtype,
      );
    });

    tearDown(() async {
      final file = File(filePath);

      if (await file.exists()) {
        await file.delete();
      }

      injector.clearAll();
    });

    test('should persist hyperparameters', () {
      expect(regressor.optimizerType, optimizerType);
      expect(regressor.learningRateType, learningRateType);
      expect(regressor.initialCoefficientsType, initialCoefficientsType);
      expect(regressor.initialLearningRate, initialLearningRate);
      expect(regressor.minCoefficientsUpdate, minCoefficientsUpdate);
      expect(regressor.lambda, lambda);
      expect(regressor.regularizationType, regularizationType);
      expect(regressor.randomSeed, randomSeed);
      expect(regressor.batchSize, batchSize);
      expect(regressor.initialCoefficients, initialCoefficients);
      expect(regressor.isFittingDataNormalized, isFittingDataNormalized);
      expect(regressor.iterationsLimit, iterationsLimit);
    });

    test('should serialize', () {
      final encoded = regressor.toJson();

      expect(encoded, {
        linearRegressorOptimizerTypeJsonKey:
          gradientLinearOptimizerTypeEncodedValue,
        linearRegressorLearningRateTypeJsonKey:
          learningRateTypeToEncodedValue[LearningRateType.decreasingAdaptive],
        linearRegressorInitialCoefficientsTypeJsonKey:
          zeroesInitialCoefficientsTypeJsonEncodedValue,
        linearRegressorInitialLearningRateTypeJsonKey: initialLearningRate,
        linearRegressorMinCoefficientsUpdateJsonKey: minCoefficientsUpdate,
        linearRegressorLambdaJsonKey: lambda,
        linearRegressorRegularizationTypeJsonKey:
          l2RegularizationTypeJsonEncodedValue,
        linearRegressorRandomSeedJsonKey: randomSeed,
        linearRegressorBatchSizeJsonKey: batchSize,
        linearRegressorInitialCoefficientsJsonKey: initialCoefficients.toJson(),
        linearRegressorFittingDataNormalizedFlagJsonKey:
          isFittingDataNormalized,
        linearRegressorTargetNameJsonKey: targetName,
        linearRegressorFitInterceptJsonKey: fitIntercept,
        linearRegressorInterceptScaleJsonKey: interceptScale,
        linearRegressorCoefficientsJsonKey: regressor.coefficients.toJson(),
        linearRegressorDTypeJsonKey: dTypeToJson(dtype),
        linearRegressorCostPerIterationJsonKey: regressor.costPerIteration,
        linearRegressorIterationsLimitJsonKey: regressor.iterationsLimit,
        jsonSchemaVersionJsonKey: linearRegressorJsonSchemaVersion,
      });
    });

    test('should return a valid pointer to a file while saving as json', () async {
      final file = await regressor.saveAsJson(filePath);

      expect(await file.exists(), isTrue);
      expect(file.path, filePath);
    });

    test('should save json-serialized model to file', () async {
      await regressor.saveAsJson(filePath);

      final file = File(filePath);

      expect(await file.exists(), isTrue);
      expect(file.path, filePath);
    });

    test('should restore from json file', () async {
      await regressor.saveAsJson(filePath);

      final file = File(filePath);
      final json = await file.readAsString();
      final restoredModel = LinearRegressor.fromJson(json);

      expect(restoredModel.dtype, regressor.dtype);
      expect(restoredModel.interceptScale, regressor.interceptScale);
      expect(restoredModel.fitIntercept, regressor.fitIntercept);
      expect(restoredModel.coefficients, regressor.coefficients);
      expect(restoredModel.targetName, regressor.targetName);
      expect(restoredModel.costPerIteration, regressor.costPerIteration);
    });
  });
}
