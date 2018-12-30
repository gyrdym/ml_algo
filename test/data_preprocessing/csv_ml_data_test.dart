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

  final data = Float32x4CsvMLDataInternal.fromFile(fileName, labelIdx, columnsToRead: columnsToRead);
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

Future testCsvWithCategories({String fileName, int labelIdx, int rowNum, Map<String, List<Object>> categories,
  List<Tuple2<int, int>> columnsToRead,
  void testContentFn(MLMatrix<Float32x4> features, MLVector<Float32x4> labels, List<String> headers)}) async {

  final data = Float32x4CsvMLDataInternal.fromFile(fileName, labelIdx, columnsToRead: columnsToRead,
      categories: categories);
  final header = await data.header;
  final features = await data.features;
  final labels = await data.labels;

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

    test('should encode categorical data via provided encoder', () async {
      await testCsvWithCategories(
          fileName: 'test/data_preprocessing/data/elo_blatter.csv',
          labelIdx: 1,
          rowNum: 209,
          columnsToRead: [
            const Tuple2(1, 7),
          ],
          categories: {
            'confederation': ['CAF', 'UEFA', 'AFC', 'CONCACAF', 'CONMEBOL', 'OFC'],
            'gdp_source': [
              'World Bank',
              'CIA (2005)',
              'CIA (2004)',
              'IMF',
              'World Bank\'s estimate for UK, adjusted per http://en.wikipedia.org/w/index.php?title=Countries_of_the_United_Kingdom_by_GVA_per_capita&oldid=153382497',
              'CIA (2008)',
              'CIA (2007)',
              'CIA',
              'South Sudan not independent in 2006; used Sudan\'s figure instead',
              'CIA (2004, French Polynesia)',
            ],
            'popu_source': [
              '2007: http://en.wikipedia.org/wiki/Tahiti',
              'CIA (2007)',
              'http://en.wikipedia.org/wiki/Demographics_of_Northern_Ireland#Population',
              'http://en.wikipedia.org/wiki/Demographics_of_Scotland',
              'http://en.wikipedia.org/wiki/Demographics_of_Wales#Population',
              'http://en.wikipedia.org/wiki/Demography_of_England#Population',
              'IMF',
              'World Bank',
            ]
          },
          testContentFn: (features, labels, header) {
            expect(header, equals([
              'elo98', 'elo15', 'confederation', 'gdp06', 'popu06', 'gdp_source', 'popu_source',
            ]));

            expect(features.getRow(0), floatIterableAlmostEqualTo(<double>[
              1116.0, // elo15
              0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // confederation
              1076.461378, // gdp06
              25631282.0, // popu06
              1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // gdp_source
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, // popu_source
            ]));

            expect(features.getRow(5), floatIterableAlmostEqualTo(<double>[
              641.0, // elo15
              0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // confederation
              8800.0, // gdp06
              13677.0, // popu06
              0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // gdp_source
              0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // popu_source
            ]));
          }
      );
    });

    test('should throw an error if label index is not in provided ranges', () async {
      expect(() =>
          Float32x4CsvMLDataInternal.fromFile(
            'test/data_preprocessing/data/elo_blatter.csv',
            1,
            columnsToRead: [const Tuple2(2, 3), const Tuple2(5, 7)],
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
  });
}