import 'dart:async';

import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/csv_data_frame.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/validator/params_validator.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:test/test.dart';
import 'package:tuple/tuple.dart';

import '../../test_utils/mocks.dart';

Future testCsvData(
    {String fileName,
    int labelIdx,
    int expectedColsNum,
    int expectedRowsNum,
    List<Tuple2<int, int>> rows,
    List<Tuple2<int, int>> columns,
    CategoricalDataEncoderFactory categoricalDataFactoryMock,
    DataFrameParamsValidator validatorMock,
    void testContentFn(
        Matrix features, Matrix labels, List<String> headers)}) async {
  categoricalDataFactoryMock ??= createCategoricalDataEncoderFactoryMock();
  validatorMock ??=
      createDataFrameParamsValidatorMock(validationShouldBeFailed: false);

  final data = CsvDataFrame.fromFile(
    fileName,
    labelIdx: labelIdx,
    columns: columns,
    rows: rows,
    encoderFactory: categoricalDataFactoryMock,
    paramsValidator: validatorMock,
  );
  final header = await data.header;
  final features = await data.features;
  final labels = await data.labels;

  if (columns == null) {
    expect(header.length, equals(expectedColsNum + 1));
    expect(features.columnsNum, equals(expectedColsNum));
  }

  expect(features.rowsNum, equals(expectedRowsNum));
  expect(labels.rowsNum, equals(expectedRowsNum));

  testContentFn(features, labels, header);
}
