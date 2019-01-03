import 'dart:async';
import 'dart:typed_data';

import 'package:ml_algo/src/data_preprocessing/ml_data/csv_ml_data.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:test/test.dart';
import 'package:tuple/tuple.dart';

import '../unit_test_helpers/floating_point_iterable_matchers.dart';

Future testCsvWithCategories({String fileName, int labelIdx, int rowNum, Map<String, List<Object>> categories,
  List<Tuple2<int, int>> columnsToRead,
  void testContentFn(MLMatrix<Float32x4> features, MLVector<Float32x4> labels, List<String> headers)}) async {

  final data = Float32x4CsvMLDataInternal.fromFile(fileName, labelIdx: labelIdx, columns: columnsToRead,
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
              1076.461425, // gdp06
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
  });
}
