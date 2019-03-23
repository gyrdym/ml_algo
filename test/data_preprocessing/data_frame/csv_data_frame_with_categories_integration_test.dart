import 'dart:async';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/csv_data_frame.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';
import 'package:tuple/tuple.dart';

Future testCsvWithCategories(
    {String fileName,
    bool headerExist = true,
    int labelIdx,
    int rowNum,
    List<Tuple2<int, int>> columns,
    Map<String, CategoricalDataEncoderType> categoryNameToEncoder,
    Map<int, CategoricalDataEncoderType> categoryIndexToEncoder,
    void testContentFn(Matrix features, Matrix labels, List<String> headers,
        DataFrame dataFrame)}) async {
  final dataFrame = CsvDataFrame.fromFile(fileName,
      labelIdx: labelIdx,
      columns: columns,
      headerExists: headerExist,
      categoryNameToEncoder: categoryNameToEncoder,
      categoryIndexToEncoder: categoryIndexToEncoder,
  );
  final header = await dataFrame.header;
  final features = await dataFrame.features;
  final labels = await dataFrame.labels;

  expect(features.rowsNum, equals(rowNum));
  expect(labels.rowsNum, equals(rowNum));

  testContentFn(features, labels, header, dataFrame);
}

void main() {
  group('CsvDataFrame', () {
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
          testContentFn: (features, labels, header, dataFrame) {
            expect(header,
                equals(['feature_1', 'feature_2', 'feature_3', 'score']));
            expect(
                features,
                equals([
                  //  feature 1     feature 2      feature 3
                  [1.0, 0.0, 0.0, /**/ 0.0, /**/ 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, /**/ 1.0, /**/ 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, /**/ 2.0, /**/ 0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0, /**/ 3.0, /**/ 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, /**/ 4.0, /**/ 0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0, /**/ 5.0, /**/ 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, /**/ 0.0, /**/ 0.0, 0.0, 1.0],
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

    test('should encode categorical data (`categorIndexToEncoder` parameter)',
            () async {
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
          testContentFn: (features, labels, header, dataFrame) {
            expect(header,
                equals(['feature_1', 'feature_2', 'feature_3', 'score']));
            expect(
                features,
                equals([
                  [1.0, 0.0, 0.0, /**/ 0.0, /**/ 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, /**/ 1.0, /**/ 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, /**/ 2.0, /**/ 0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0, /**/ 3.0, /**/ 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, /**/ 4.0, /**/ 0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0, /**/ 5.0, /**/ 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, /**/ 0.0, /**/ 0.0, 0.0, 1.0],
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
          testContentFn: (features, labels, header, dataFrame) {
            expect(header, isNull);
            expect(
                features,
                equals([
                  [1.0, 0.0, 0.0, /**/ 0.0, /**/ 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, /**/ 1.0, /**/ 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, /**/ 2.0, /**/ 0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0, /**/ 3.0, /**/ 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, /**/ 4.0, /**/ 0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0, /**/ 5.0, /**/ 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, /**/ 0.0, /**/ 0.0, 0.0, 1.0],
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

    test('should decode categorical data', () async {
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
          testContentFn: (features, labels, header, dataFrame) {
            final decoded = dataFrame.decode(Matrix.from([
              [1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0],
              [0.0, 1.0, 0.0],
              [0.0, 1.0, 0.0],
              [1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0],
            ]), colName: 'feature_1');
            expect(decoded, equals(['value_1_1', 'value_1_2', 'value_1_3',
            'value_1_2', 'value_1_2', 'value_1_1', 'value_1_3']));
          });
    });

    test('should throw an exception if one tries to decode a data without '
        'providing column name or column index', () async {
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
          testContentFn: (features, labels, header, dataFrame) {
            expect(() => dataFrame.decode(Matrix.from([
              [1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
            ])), throwsException);
          });
    });

    test('should throw an exception if one tries to decode a data from headless'
        'dataset providing just a column name', () async {
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
          testContentFn: (features, labels, header, dataFrame) {
            expect(() => dataFrame.decode(Matrix.from([
              [1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
            ]), colName: 'feature_1'), throwsException);
          });
    });

    test('should throw an exception if one tries to decode a data providing '
        'inexistent column name', () async {
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
          testContentFn: (features, labels, header, dataFrame) {
            expect(() => dataFrame.decode(Matrix.from([
              [1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
            ]), colName: 'country'), throwsException);
          });
    });

    test('should throw a range error if one tries to decode a data providing '
        'improper column index', () async {
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
          testContentFn: (features, labels, header, dataFrame) {
            expect(() => dataFrame.decode(Matrix.from([
              [1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
            ]), colIdx: 45), throwsRangeError);
          });
    });

    test('should throw an exception if one tries to decode a data providing '
        'index of non-categorical column', () async {
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
          testContentFn: (features, labels, header, dataFrame) {
            expect(() => dataFrame.decode(Matrix.from([
              [1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
            ]), colIdx: 3), throwsException);
          });
    });
  });
}
