import 'dart:async';
import 'dart:typed_data';

import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/float32x4_csv_ml_data.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/validator/ml_data_params_validator.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';
import 'package:tuple/tuple.dart';

import '../../test_utils/mocks.dart';

Future testCsvData({
  String fileName,
  int labelIdx,
  int expectedColsNum,
  int expectedRowsNum,
  List<Tuple2<int, int>> rows,
  List<Tuple2<int, int>> columns,
  CategoricalDataEncoderFactory categoricalDataFactoryMock,
  MLDataParamsValidator validatorMock,
  void testContentFn(MLMatrix<Float32x4> features, MLVector<Float32x4> labels, List<String> headers)}) async {

  categoricalDataFactoryMock ??= createCategoricalDataEncoderFactoryMock();
  validatorMock ??= createMLDataParamsValidatorMock(validationShouldBeFailed: false);

  final data = Float32x4CsvMLDataInternal.fromFile(fileName,
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
  expect(labels.length, equals(expectedRowsNum));

  testContentFn(features, labels, header);
}
