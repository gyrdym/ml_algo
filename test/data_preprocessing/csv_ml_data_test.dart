import 'dart:async';
import 'dart:typed_data';

import 'package:ml_algo/src/data_preprocessing/ml_data/csv_ml_data.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';
import 'package:tuple/tuple.dart';

import '../unit_test_helpers/float_iterable_almost_equal_to.dart';

Future testCsvWithoutCategories({String fileName, int labelIdx, int colNum, int rowNum,
  List<Tuple2<int, int>> columnsToRead,
  void testContentFn(MLMatrix<Float32x4> features, MLVector<Float32x4> labels, List<String> headers)}) async {

  final data = Float32x4CsvMLDataInternal.fromFile(fileName, labelIdx: labelIdx, columns: columnsToRead);
  final header = await data.header;
  final features = await data.features;
  final labels = await data.labels;

  if (columnsToRead == null) {
    expect(header.length, equals(colNum + 1));
    expect(features.columnsNum, equals(colNum));
  }

  expect(features.rowsNum, equals(rowNum));
  expect(labels.length, equals(rowNum));

  testContentFn(features, labels, header);
}

void main() {
  group('CsvMLData', () {
    test('should properly parse csv file', () async {
      await testCsvWithoutCategories(
        fileName: 'test/data_preprocessing/data/pima_indians_diabetes_database.csv',
        labelIdx: 8,
        colNum: 8,
        rowNum: 768,
        testContentFn: (features, labels, header) {
          expect(features.getRow(0), floatIterableAlmostEqualTo([6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0]));
          expect(features.getRow(34), floatIterableAlmostEqualTo([10.0, 122.0, 78.0, 31.0, 0.0, 27.6, 0.512, 45.0]));
          expect([labels[0], labels[34], labels[63]], equals([1, 0, 0]));
        }
      );
    });

    test('should parse csv file with specified label column position', () async {
      await testCsvWithoutCategories(
          fileName: 'test/data_preprocessing/data/pima_indians_diabetes_database.csv',
          labelIdx: 1,
          colNum: 8,
          rowNum: 768,
          testContentFn: (features, labels, header) {
            expect(features.getRow(0), floatIterableAlmostEqualTo([6.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0, 1.0]));
            expect(features.getRow(34), floatIterableAlmostEqualTo([10.0, 78.0, 31.0, 0.0, 27.6, 0.512, 45.0, 0.0]));
            expect([labels[0], labels[34], labels[63]], equals([148.0, 122.0, 141.0]));
          }
      );
    });

    test('should parse csv file with specified label column position, position is 0', () async {
      await testCsvWithoutCategories(
          fileName: 'test/data_preprocessing/data/pima_indians_diabetes_database.csv',
          labelIdx: 0,
          colNum: 8,
          rowNum: 768,
          testContentFn: (features, labels, header) {
            expect(features.getRow(0), floatIterableAlmostEqualTo([148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0, 1.0]));
            expect(features.getRow(34), floatIterableAlmostEqualTo([122.0, 78.0, 31.0, 0.0, 27.6, 0.512, 45.0, 0.0]));
            expect([labels[0], labels[34], labels[63]], equals([6.0, 10.0, 2.0]));
          }
      );
    });

    test('should extract header data if the latter is specified', () async {
      await testCsvWithoutCategories(
          fileName: 'test/data_preprocessing/data/pima_indians_diabetes_database.csv',
          labelIdx: 0,
          colNum: 8,
          rowNum: 768,
          testContentFn: (features, labels, header) {
        expect(header, equals([
          'number of times pregnant',
          'plasma glucose concentration a 2 hours in an oral glucose tolerance test',
          'diastolic blood pressure (mm Hg)',
          'triceps skin fold thickness (mm)',
          '2-Hour serum insulin (mu U/ml)',
          'body mass index (weight in kg/(height in m)^2)',
          'diabetes pedigree function',
          'age (years)',
          'class variable (0 or 1)',
        ]));
      }
      );
    });

    test('should throw an error if label index is not in provided ranges', () async {
      expect(() =>
          Float32x4CsvMLDataInternal.fromFile(
            'test/data_preprocessing/data/elo_blatter.csv',
            labelIdx: 1,
            columns: [const Tuple2(2, 3), const Tuple2(5, 7)],
          ),
          throwsException,
      );
    });

    test('should cut out selected columns', () async {
      await testCsvWithoutCategories(
          fileName: 'test/data_preprocessing/data/pima_indians_diabetes_database.csv',
          labelIdx: 8,
          colNum: 8,
          rowNum: 768,
          columnsToRead: [const Tuple2(0, 1), const Tuple2(2, 2), const Tuple2(3, 4), const Tuple2(6, 8)],
          testContentFn: (features, labels, header) {
            expect(features.getRow(0), floatIterableAlmostEqualTo([6.0, 148.0, 72.0, 35.0, 0.0, 0.627, 50.0]));
            expect(features.getRow(34), floatIterableAlmostEqualTo([10.0, 122.0, 78.0, 31.0, 0.0, 0.512, 45.0]));
            expect([labels[0], labels[34], labels[63]], equals([1, 0, 0]));
          }
      );
    });

    test('should throw an error if there are intersecting column ranges while reading selected columns', () {
      final actual = () => Float32x4CsvMLDataInternal.fromFile(
          'test/data_preprocessing/data/pima_indians_diabetes_database.csv',
          labelIdx: 8,
          columns: [
            const Tuple2(0, 1), // first and
            const Tuple2(1, 2), // second ranges are intersecting
            const Tuple2(3, 4),
            const Tuple2(6, 8)],
      );
      expect(actual, throwsException);
    });

    test('should throw an error if label index with null value passed to constructor', () {
      final actual = () => Float32x4CsvMLDataInternal.fromFile(
        'test/data_preprocessing/data/pima_indians_diabetes_database.csv',
        labelIdx: null,
        columns: [
          const Tuple2(0, 1),
          const Tuple2(2, 2),
          const Tuple2(3, 4),
          const Tuple2(6, 8)],
      );
      expect(actual, throwsException);
    });
  });
}