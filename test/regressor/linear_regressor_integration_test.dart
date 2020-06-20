import 'dart:io';

import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_json_keys.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/dtype_to_json.dart';
import 'package:test/test.dart';

void main() {
  group('LinearRegressor', () {
    final featureNames = ['feature 1', 'feature 2', 'feature 3'];
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

    LinearRegressor regressor;

    setUp(() {
      regressor = LinearRegressor(dataSet, targetName,
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
    });

    test('should serialize', () {
      final encoded = regressor.toJson();

      expect(encoded, {
        linearRegressorTargetNameJsonKey: targetName,
        linearRegressorFitInterceptJsonKey: fitIntercept,
        linearRegressorInterceptScaleJsonKey: interceptScale,
        linearRegressorCoefficientsJsonKey: regressor.coefficients.toJson(),
        linearRegressorDTypeJsonKey: dTypeToJson(dtype),
        linearRegressorCostPerIterationJsonKey: regressor.costPerIteration,
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

      print(regressor.costPerIteration);

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
