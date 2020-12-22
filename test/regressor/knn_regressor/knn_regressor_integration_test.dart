import 'dart:io';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type_json_keys.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_constants.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_json_keys.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_constants.dart';
import 'package:ml_algo/src/regressor/knn_regressor/knn_regressor_json_keys.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/dtype_to_json.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';

import '../../helpers.dart';

void main() {
  group('KnnRegressor', () {
    final k = 2;
    final data = DataFrame(
      <Iterable<num>>[
        [20, 20, 20, 20, 20, 1],
        [30, 30, 30, 30, 30, 2],
        [15, 15, 15, 15, 15, 3],
        [25, 25, 25, 25, 25, 4],
        [10, 10, 10, 10, 10, 5],
      ],
      header: ['first', 'second', 'third', 'fourth', 'fifth', 'target'],
      headerExists: false,
    );
    final testFeatures = Matrix.fromList([
      [9.0, 9.0, 9.0, 9.0, 9.0],
    ]);
    final targetName = 'target';

    group('prediction', () {
      test('should predict values using uniform kernel', () {
        final regressor = KnnRegressor(
          data,
          targetName,
          k,
          kernel: KernelType.uniform,
          distance: Distance.euclidean,
        );

        final actual = regressor.predict(
          DataFrame.fromMatrix(testFeatures),
        );

        expect(actual.header, equals(['target']));
        expect(actual.toMatrix(), equals([[4.0]]));
      });

      test('should predict values using epanechnikov kernel', () {
        final regressor = KnnRegressor(
            data,
            targetName,
            k,
            kernel: KernelType.epanechnikov);

        final actual = regressor.predict(
          DataFrame.fromMatrix(testFeatures),
        );

        expect(actual.header, equals(['target']));
        expect(actual.toMatrix(), iterable2dAlmostEqualTo([[5.0]]));
      });
    });

    group('toJson', () {
      test('should return json-encoded object', () {
        final regressor = KnnRegressor(
            data,
            targetName,
            k,
            distance: Distance.manhattan,
            dtype: DType.float64,
            kernel: KernelType.gaussian);

        expect(regressor.toJson(), {
          knnRegressorDTypeJsonKey: dTypeToJson(DType.float64),
          knnRegressorTargetNameJsonKey: targetName,
          knnRegressorSolverJsonKey: {
            knnSolverTrainFeaturesJsonKey: matrixToJson(
              Matrix.fromList([
                [20, 20, 20, 20, 20],
                [30, 30, 30, 30, 30],
                [15, 15, 15, 15, 15],
                [25, 25, 25, 25, 25],
                [10, 10, 10, 10, 10],
              ], dtype: DType.float64),
            ),
            knnSolverTrainOutcomesJsonKey: matrixToJson(
              Matrix.fromList([
                [1],
                [2],
                [3],
                [4],
                [5],
              ], dtype: DType.float64)
            ),
            knnSolverKJsonKey: k,
            knnSolverDistanceTypeJsonKey: distanceTypeToJson(Distance.manhattan),
            knnSolverStandardizeJsonKey: true,
            jsonSchemaVersionJsonKey: knnSolverJsonSchemaVersion,
          },
          knnRegressorKernelJsonKey: gaussianKernelEncodedValue,
          jsonSchemaVersionJsonKey: knnRegressorJsonSchemaVersion,
        });
      });
    });

    group('saveAsJson', () {
      final testFileName = 'test/regressor/knn_regressor/serialized_regressor.json';
      final regressor = KnnRegressor(
        data,
        targetName,
        k,
        kernel: KernelType.cosine,
        distance: Distance.hamming,
        dtype: DType.float64,
      );

      tearDown(() async {
        final file = await File(testFileName);
        if (!await file.exists()) {
          return;
        }
        await file.delete();
      });

      test('should save to file as json', () async {
        await regressor.saveAsJson(testFileName);

        final file = await File(testFileName);
        final fileExists = await file.exists();

        expect(fileExists, isTrue);
      });

      test('should save to a restorable json', () async {
        await regressor.saveAsJson(testFileName);

        final file = await File(testFileName);
        final json = await file.readAsString();
        final restoredRegressor = KnnRegressor.fromJson(json);

        expect(restoredRegressor.toJson(), regressor.toJson());
      });
    });
  });
}
