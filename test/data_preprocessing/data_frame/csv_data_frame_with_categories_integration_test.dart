import 'dart:async';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/csv_data_frame.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';
import 'package:tuple/tuple.dart';

import '../../test_utils/helpers/floating_point_iterable_matchers.dart';

Future testCsvWithCategories(
    {String fileName,
    bool headerExist = true,
    int labelIdx,
    int rowNum,
    List<Tuple2<int, int>> columns,
    Map<String, CategoricalDataEncoderType> categoryNameToEncoder,
    Map<int, CategoricalDataEncoderType> categoryIndexToEncoder,
    void testContentFn(
        Matrix features, Matrix labels, List<String> headers)}) async {
  final data = CsvDataFrame.fromFile(fileName,
      labelIdx: labelIdx,
      columns: columns,
      headerExists: headerExist,
      categoryNameToEncoder: categoryNameToEncoder,
      categoryIndexToEncoder: categoryIndexToEncoder,
  );
  final header = await data.header;
  final features = await data.features;
  final labels = await data.labels;

  expect(features.rowsNum, equals(rowNum));
  expect(labels.rowsNum, equals(rowNum));

  testContentFn(features, labels, header);
}

void main() {
  group('CsvDataFrame', () {
    test('should encode data with help of predefined categories (`categories` '
        'parameter)', () async {
      await testCsvWithCategories(
          fileName: 'test/data_preprocessing/test_data/elo_blatter.csv',
          labelIdx: 1,
          rowNum: 209,
          columns: [
            const Tuple2(1, 7),
          ],
          testContentFn: (features, labels, header) {
            expect(
                header,
                equals([
                  'elo98',
                  'elo15',
                  'confederation',
                  'gdp06',
                  'popu06',
                  'gdp_source',
                  'popu_source',
                ]));

            expect(
                features.getRow(0),
                vectorAlmostEqualTo(<double>[
                  1116.0, // elo15
                  0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // confederation
                  1076.461425, // gdp06
                  25631282.0, // popu06
                  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, // gdp_source
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, // popu_source
                ]));

            expect(
                features.getRow(5),
                vectorAlmostEqualTo(<double>[
                  641.0, // elo15
                  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // confederation
                  8800.0, // gdp06
                  13677.0, // popu06
                  0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, // gdp_source
                  0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // popu_source
                ]));
          });
    });

    test('should encode categorical data (`categoryNameToEncoder` parameter)',
        () async {
      await testCsvWithCategories(
          fileName: 'test/data_preprocessing/test_data/fake_data.csv',
          labelIdx: 3,
          rowNum: 7,
          columns: [
            const Tuple2(0, 3),
          ],
          categoryNameToEncoder: {
            'feature_1': CategoricalDataEncoderType.oneHot,
            'feature_2': CategoricalDataEncoderType.ordinal,
            'feature_3': CategoricalDataEncoderType.oneHot,
          },
          testContentFn: (features, labels, header) {
            expect(header,
                equals(['feature_1', 'feature_2', 'feature_3', 'score']));
            expect(
                features,
                equals([
                  [1.0, 0.0, 0.0, /**/ 1.0, /**/ 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, /**/ 2.0, /**/ 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, /**/ 3.0, /**/ 0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0, /**/ 4.0, /**/ 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, /**/ 5.0, /**/ 0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0, /**/ 6.0, /**/ 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, /**/ 1.0, /**/ 0.0, 0.0, 1.0],
                ]));
            expect(labels, equals([
              [1],
              [10],
              [200],
              [300],
              [400],
              [500],
              [700]
            ]));
          });
    });

    test('should encode categorical data (`categorIndexToEncoder` '
        'parameter)', () async {
      await testCsvWithCategories(
          fileName: 'test/data_preprocessing/test_data/fake_data.csv',
          labelIdx: 3,
          rowNum: 7,
          columns: [
            const Tuple2(0, 3),
          ],
          categoryIndexToEncoder: {
            0: CategoricalDataEncoderType.oneHot,
            1: CategoricalDataEncoderType.ordinal,
            2: CategoricalDataEncoderType.oneHot,
          },
          testContentFn: (features, labels, header) {
            expect(header,
                equals(['feature_1', 'feature_2', 'feature_3', 'score']));
            expect(
                features,
                equals([
                  [1.0, 0.0, 0.0, /**/ 1.0, /**/ 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, /**/ 2.0, /**/ 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, /**/ 3.0, /**/ 0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0, /**/ 4.0, /**/ 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, /**/ 5.0, /**/ 0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0, /**/ 6.0, /**/ 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, /**/ 1.0, /**/ 0.0, 0.0, 1.0],
                ]));
            expect(labels, equals([
              [1],
              [10],
              [200],
              [300],
              [400],
              [500],
              [700]
            ]));
          });
    });

    test('should encode categorical data in headless dataset', () async {
      await testCsvWithCategories(
          fileName: 'test/data_preprocessing/test_data/fake_data_headless.csv',
          headerExist: false,
          labelIdx: 3,
          rowNum: 7,
          columns: [
            const Tuple2(0, 3),
          ],
          categoryIndexToEncoder: {
            0: CategoricalDataEncoderType.oneHot,
            1: CategoricalDataEncoderType.ordinal,
            2: CategoricalDataEncoderType.oneHot,
          },
          testContentFn: (features, labels, header) {
            expect(header, isNull);
            expect(
                features,
                equals([
                  [1.0, 0.0, 0.0, /**/ 1.0, /**/ 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, /**/ 2.0, /**/ 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, /**/ 3.0, /**/ 0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0, /**/ 4.0, /**/ 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, /**/ 5.0, /**/ 0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0, /**/ 6.0, /**/ 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, /**/ 1.0, /**/ 0.0, 0.0, 1.0],
                ]));
            expect(labels, equals([
              [1],
              [10],
              [200],
              [300],
              [400],
              [500],
              [700]
            ]));
          });
    });
  });
}
