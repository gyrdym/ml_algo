import 'dart:io';

import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier.dart';
import 'package:ml_algo/src/classifier/knn_classifier/knn_classifier_json_keys.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type.dart';
import 'package:ml_algo/src/knn_kernel/kernel_type_json_keys.dart';
import 'package:ml_algo/src/knn_solver/knn_solver_json_keys.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/dtype_to_json.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

void main() {
  group('KnnClassifier', () {
    final targetName = 'Outcome';
    final classLabel1 = 10.0;
    final classLabel2 = 20.0;
    final classLabel3 = 30.0;
    final header = ['feature_1', 'feature_2', 'feature_3', targetName];
    final trainData = DataFrame(
      [
        [123,   321,  444, classLabel2],
        [  2,    11,  -10, classLabel1],
        [340, -1002,   93, classLabel3],
        [ 17,  2219, 3019, classLabel1],
        [  7,     0,   -1, classLabel2],
        [ 13,    31,   44, classLabel3],
      ],
      headerExists: false,
      header: header,
    );
    final classLabelPrefix = 'Awesome class';
    final k = 3;

    group('toJson', () {
      test('should return json-encoded classifier', () {
        final classifier = KnnClassifier(
          trainData,
          targetName,
          k,
          kernel: KernelType.cosine,
          distance: Distance.hamming,
          classLabelPrefix: classLabelPrefix,
          dtype: DType.float64,
        );

        expect(classifier.toJson(), {
          knnClassifierTargetColumnNameJsonKey: targetName,
          knnClassifierDTypeJsonKey: dTypeToJson(DType.float64),
          knnClassifierClassLabelsJsonKey: [classLabel2, classLabel1, classLabel3],
          knnClassifierKernelJsonKey: cosineKernelEncodedValue,
          knnClassifierSolverJsonKey: {
            knnSolverTrainFeaturesJsonKey:
              Matrix.fromList([
                [123,   321,  444],
                [  2,    11,  -10],
                [340, -1002,   93],
                [ 17,  2219, 3019],
                [  7,     0,   -1],
                [ 13,    31,   44],
              ], dtype: DType.float64).toJson(),
            knnSolverTrainOutcomesJsonKey:
                Matrix.fromList([
                  [classLabel2],
                  [classLabel1],
                  [classLabel3],
                  [classLabel1],
                  [classLabel2],
                  [classLabel3],
                ], dtype: DType.float64).toJson(),
            knnSolverKJsonKey: k,
            knnSolverDistanceTypeJsonKey: distanceTypeToJson(Distance.hamming),
            knnSolverStandardizeJsonKey: true,
          },
          knnClassifierClassLabelPrefixJsonKey: classLabelPrefix,
        });
      });
    });

    group('saveAsJson', () {
      final testFileName = 'test/classifier/knn_classifier/serialized_classifier.json';
      final classifier = KnnClassifier(
        trainData,
        targetName,
        k,
        kernel: KernelType.cosine,
        distance: Distance.hamming,
        classLabelPrefix: classLabelPrefix,
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
        await classifier.saveAsJson(testFileName);

        final file = await File(testFileName);
        final fileExists = await file.exists();

        expect(fileExists, isTrue);
      });

      test('should save to a restorable json', () async {
        await classifier.saveAsJson(testFileName);

        final file = await File(testFileName);
        final json = await file.readAsString();
        final restoredClassifier = KnnClassifier.fromJson(json);

        expect(restoredClassifier.toJson(), classifier.toJson());
      });
    });
  });
}
