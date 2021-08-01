import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/regressor/linear_regressor/_injector.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';
import '../../mocks.mocks.dart';

void main() {
  group('LinearRegressor', () {
    final features = [
      [23, 32, 33],
      [11, -1, 0],
      [87, 77, 39],
    ];
    final outcomes = [
      [100],
      [200],
      [300],
    ];
    final featuresName = ['f1', 'f2', 'f3'];
    final targetName = 'target';
    final header = [...featuresName, targetName];
    final fittingData = DataFrame([
      [...features[0], ...outcomes[0]],
      [...features[1], ...outcomes[1]],
      [...features[2], ...outcomes[2]],
    ], headerExists: false, header: header);
    final optimizerType = LinearOptimizerType.coordinate;
    final iterationsLimit = 199;
    final learningRateType = LearningRateType.constant;
    final initialCoefficientsType = InitialCoefficientsType.zeroes;
    final initialLearningRate = 0.7;
    final minCoefficientsUpdate = 1.57;
    final lambda = 125.4;
    final regularizationType = RegularizationType.L1;
    final fitIntercept = false;
    final interceptScale = 781.999;
    final randomSeed = 4561;
    final batchSize = 2;
    final initialCoeffs = Matrix.column([201, 301, 401, 501]);
    final isFittingDataNormalized = true;
    final collectLearningDate = true;
    final dtype = DType.float32;
    final regressorMock = MockLinearRegressor();
    final factoryMock = createLinearRegressorFactoryMock(regressorMock);

    late LinearRegressor regressor;

    setUp(() {
      linearRegressorInjector.registerSingleton(() => factoryMock);

      regressor = LinearRegressor(
        fittingData,
        targetName,
        optimizerType: optimizerType,
        iterationsLimit: iterationsLimit,
        learningRateType: learningRateType,
        initialCoefficientsType: initialCoefficientsType,
        initialLearningRate: initialLearningRate,
        minCoefficientsUpdate: minCoefficientsUpdate,
        lambda: lambda,
        regularizationType: regularizationType,
        fitIntercept: fitIntercept,
        interceptScale: interceptScale,
        randomSeed: randomSeed,
        batchSize: batchSize,
        initialCoefficients: initialCoeffs,
        isFittingDataNormalized: isFittingDataNormalized,
        collectLearningData: collectLearningDate,
        dtype: dtype,
      );
    });

    tearDown(() {
      injector.clearAll();
      linearRegressorInjector.clearAll();
    });

    test('should call the factory', () {
      verify(factoryMock.create(
        fittingData: fittingData,
        targetName: targetName,
        optimizerType: optimizerType,
        iterationsLimit: iterationsLimit,
        learningRateType: learningRateType,
        initialCoefficientsType: initialCoefficientsType,
        initialLearningRate: initialLearningRate,
        minCoefficientsUpdate: minCoefficientsUpdate,
        lambda: lambda,
        regularizationType: regularizationType,
        fitIntercept: fitIntercept,
        interceptScale: interceptScale,
        randomSeed: randomSeed,
        batchSize: batchSize,
        initialCoefficients: initialCoeffs,
        isFittingDataNormalized: isFittingDataNormalized,
        collectLearningData: collectLearningDate,
        dtype: dtype,
      )).called(1);
    });

    test('should return an instance from the factory', () {
      expect(regressor, same(regressorMock));
    });
  });
}
