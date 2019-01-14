import 'package:ml_algo/src/data_preprocessing/ml_data/float32x4_csv_ml_data.dart';
import 'package:test/test.dart';
import 'package:tuple/tuple.dart';

import '../test_utils/helpers/floating_point_iterable_matchers.dart';
import '../test_utils/mocks.dart';
import 'test_helpers/test_csv_data.dart';

void main() {
  group('Float32x4CsvMLData (categories-less)', () {
    test('should properly parse csv file', () async {
      await testCsvData(
        fileName: 'test/data_preprocessing/data/pima_indians_diabetes_database.csv',
        labelIdx: 8,
        expectedColsNum: 8,
        expectedRowsNum: 768,
        testContentFn: (features, labels, header) {
          expect(features.getRow(0), floatIterableAlmostEqualTo([6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0]));
          expect(features.getRow(34), floatIterableAlmostEqualTo([10.0, 122.0, 78.0, 31.0, 0.0, 27.6, 0.512, 45.0]));
          expect([labels[0], labels[34], labels[63]], equals([1, 0, 0]));
        }
      );
    });

    test('should parse csv file with specified label column position', () async {
      await testCsvData(
          fileName: 'test/data_preprocessing/data/pima_indians_diabetes_database.csv',
          labelIdx: 1,
          expectedColsNum: 8,
          expectedRowsNum: 768,
          testContentFn: (features, labels, header) {
            expect(features.getRow(0), floatIterableAlmostEqualTo([6.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0, 1.0]));
            expect(features.getRow(34), floatIterableAlmostEqualTo([10.0, 78.0, 31.0, 0.0, 27.6, 0.512, 45.0, 0.0]));
            expect([labels[0], labels[34], labels[63]], equals([148.0, 122.0, 141.0]));
          }
      );
    });

    test('should parse csv file with specified label column position, position is 0', () async {
      await testCsvData(
          fileName: 'test/data_preprocessing/data/pima_indians_diabetes_database.csv',
          labelIdx: 0,
          expectedColsNum: 8,
          expectedRowsNum: 768,
          testContentFn: (features, labels, header) {
            expect(features.getRow(0), floatIterableAlmostEqualTo([148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0, 1.0]));
            expect(features.getRow(34), floatIterableAlmostEqualTo([122.0, 78.0, 31.0, 0.0, 27.6, 0.512, 45.0, 0.0]));
            expect([labels[0], labels[34], labels[63]], equals([6.0, 10.0, 2.0]));
          }
      );
    });

    test('should extract header data if the latter is specified', () async {
      await testCsvData(
          fileName: 'test/data_preprocessing/data/pima_indians_diabetes_database.csv',
          labelIdx: 0,
          expectedColsNum: 8,
          expectedRowsNum: 768,
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
      await testCsvData(
          fileName: 'test/data_preprocessing/data/pima_indians_diabetes_database.csv',
          labelIdx: 8,
          expectedColsNum: 8,
          expectedRowsNum: 768,
          columns: [const Tuple2(0, 1), const Tuple2(2, 2), const Tuple2(3, 4), const Tuple2(6, 8)],
          testContentFn: (features, labels, header) {
            expect(features.getRow(0), floatIterableAlmostEqualTo([6.0, 148.0, 72.0, 35.0, 0.0, 0.627, 50.0]));
            expect(features.getRow(34), floatIterableAlmostEqualTo([10.0, 122.0, 78.0, 31.0, 0.0, 0.512, 45.0]));
            expect([labels[0], labels[34], labels[63]], equals([1, 0, 0]));
          }
      );
    });

    test('should throw an error if there are intersecting column ranges while parsing csv file', () {
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

    test('should cut out selected rows, all rows in one range', () async {
      await testCsvData(
          fileName: 'test/data_preprocessing/data/pima_indians_diabetes_database.csv',
          labelIdx: 8,
          rows: [const Tuple2(0, 767)],
          expectedColsNum: 8,
          expectedRowsNum: 768,
          testContentFn: (features, labels, header) {
            expect(features.getRow(0), floatIterableAlmostEqualTo([6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0]));
            expect(features.getRow(767), floatIterableAlmostEqualTo([1.0, 93.0, 70.0, 31.0, 0.0, 30.4, 0.315, 23.0]));
            expect(() => features.getRow(768), throwsRangeError);
            expect([labels[0], labels[34], labels[767]], equals([1, 0, 0]));
            expect(() => labels[768], throwsRangeError);
          }
      );
    });

    test('should cut out selected rows, several row ranges', () async {
      await testCsvData(
          fileName: 'test/data_preprocessing/data/pima_indians_diabetes_database.csv',
          labelIdx: 8,
          rows: [
            const Tuple2(0, 2),
            const Tuple2(3, 4),
            const Tuple2(10, 15),
          ],
          expectedColsNum: 8,
          expectedRowsNum: 11,
          testContentFn: (features, labels, header) {
            expect(features, matrixAlmostEqualTo([
              [6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0],
              [1.0, 85.0, 66.0, 29.0, 0.0, 26.6, 0.351, 31.0],
              [8.0, 183.0, 64.0, 0.0, 0.0, 23.3, 0.672, 32.0],
              [1.0, 89.0, 66.0, 23.0, 94.0, 28.1, 0.167, 21.0],
              [0.0, 137.0, 40.0, 35.0, 168.0, 43.1, 2.288, 33.0],
              [4.0, 110.0, 92.0, 0.0, 0.0, 37.6, 0.191, 30.0],
              [10.0, 168.0, 74.0, 0.0, 0.0, 38.0, 0.537, 34.0],
              [10.0, 139.0, 80.0, 0.0, 0.0, 27.1, 1.441, 57.0],
              [1.0, 189.0, 60.0, 23.0, 846.0, 30.1, 0.398, 59.0],
              [5.0, 166.0, 72.0, 19.0, 175.0, 25.8, 0.587, 51.0],
              [7.0, 100.0, 0.0, 0.0, 0.0, 30.0, 0.484, 32.0],
            ]));
            expect(() => features.getRow(11), throwsRangeError);
            expect(() => features.getRow(768), throwsRangeError);

            expect(labels, equals([1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1]));
            expect(() => labels[11], throwsRangeError);
            expect(() => labels[768], throwsRangeError);
          }
      );
    });

    test('should throw an error if params validation fails', () {
      final validatorMock = createMLDataParamsValidatorMock(validationShouldBeFailed: true);
      final actual = () => Float32x4CsvMLDataInternal.fromFile(
        'test/data_preprocessing/data/pima_indians_diabetes_database.csv',
        paramsValidator: validatorMock,
      );
      expect(actual, throwsException);
    });
  });
}