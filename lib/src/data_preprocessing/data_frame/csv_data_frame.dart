import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:csv/csv.dart';
import 'package:logging/logging.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encode_unknown_strategy_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/data_frame.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/encoders_processor/encoders_processor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/encoders_processor/encoders_processor_factory_impl.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/features_extractor/features_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/features_extractor/features_extractor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/features_extractor/features_extractor_factory_impl.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/header_extractor/header_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/header_extractor/header_extractor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/header_extractor/header_extractor_factory_impl.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/labels_extractor/labels_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/labels_extractor/labels_extractor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/labels_extractor/labels_extractor_factory_impl.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/read_mask_creator/read_mask_creator_factory.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/read_mask_creator/read_mask_creator_factory_impl.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/validator/params_validator.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/validator/params_validator_impl.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/value_converter/value_converter.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/value_converter/value_converter_impl.dart';
import 'package:ml_algo/src/default_parameter_values.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:tuple/tuple.dart';

class CsvDataFrame implements DataFrame {
  CsvDataFrame.fromFile(String fileName, {
      // public parameters
      Type dtype,
      String eol = '\n',
      int labelIdx,
      bool headerExists = true,
      CategoricalDataEncoderType encoderType =
          CategoricalDataEncoderType.oneHot,
      EncodeUnknownValueStrategy encodeUnknownStrategy =
          EncodeUnknownValueStrategy.throwError,
      Map<String, List<Object>> categories,
      Map<int, List<Object>> categoriesByIndexes,
      Map<String, CategoricalDataEncoderType> categoryNameToEncoder,
      Map<int, CategoricalDataEncoderType> categoryIndexToEncoder,
      List<Tuple2<int, int>> rows,
      List<Tuple2<int, int>> columns,

      // private parameters, they are hidden by the factory
      CategoricalDataEncoderFactory encoderFactory =
        const CategoricalDataEncoderFactory(),

      DataFrameParamsValidator paramsValidator =
        const DataFrameParamsValidatorImpl(),

      DataFrameValueConverter valueConverter =
        const DataFrameValueConverterImpl(),

      DataFrameHeaderExtractorFactory headerExtractorFactory =
        const DataFrameHeaderExtractorFactoryImpl(),

      DataFrameFeaturesExtractorFactory featuresExtractorFactory =
        const DataFrameFeaturesExtractorFactoryImpl(),

      DataFrameLabelsExtractorFactory labelsExtractorFactory =
        const DataFrameLabelsExtractorFactoryImpl(),

      DataFrameReadMaskCreatorFactory readMaskCreatorFactory =
        const DataFrameReadMaskCreatorFactoryImpl(),

      DataFrameEncodersProcessorFactory encodersProcessorFactory =
        const DataFrameEncodersProcessorFactoryImpl(),

      Logger logger,
    })  : _dtype = dtype ?? DefaultParameterValues.dtype,
      _csvCodec = CsvCodec(eol: eol),
      _file = File(fileName),
      _labelIdx = labelIdx,
      _headerExists = headerExists,
      _rows = rows,
      _columns = columns,
      _categories = categories ?? {},
      _nameToEncoderType = categoryNameToEncoder ?? {},
      _indexToEncoderType = categoryIndexToEncoder ?? {},
      _encoderFactory = encoderFactory,
      _fallbackEncoderType = encoderType,
      _paramsValidator = paramsValidator,
      _valueConverter = valueConverter,
      _headerExtractorFactory = headerExtractorFactory,
      _featuresExtractorFactory = featuresExtractorFactory,
      _labelsExtractorFactory = labelsExtractorFactory,
      _readMaskCreatorFactory = readMaskCreatorFactory,
      _encodersProcessorFactory = encodersProcessorFactory,
      _logger = logger ?? Logger(_loggerPrefix) {
    final errorMsg = _paramsValidator.validate(
      labelIdx: labelIdx,
      rows: rows,
      columns: columns,
      headerExists: headerExists,
      predefinedCategories: categories,
      namesToEncoders: categoryNameToEncoder,
      indexToEncoder: categoryIndexToEncoder,
    );
    if (errorMsg.isNotEmpty) {
      throw Exception(_wrapErrorMessage(errorMsg));
    }
  }

  final Type _dtype;
  final CsvCodec _csvCodec;
  final File _file;
  final int _labelIdx;
  final bool _headerExists;
  final List<Tuple2<int, int>> _rows;
  final List<Tuple2<int, int>> _columns;
  final CategoricalDataEncoderFactory _encoderFactory;
  final DataFrameParamsValidator _paramsValidator;
  final DataFrameValueConverter _valueConverter;
  final DataFrameReadMaskCreatorFactory _readMaskCreatorFactory;
  final DataFrameHeaderExtractorFactory _headerExtractorFactory;
  final DataFrameFeaturesExtractorFactory _featuresExtractorFactory;
  final DataFrameLabelsExtractorFactory _labelsExtractorFactory;
  final DataFrameEncodersProcessorFactory _encodersProcessorFactory;
  final Logger _logger;

  final Map<String, CategoricalDataEncoderType> _nameToEncoderType;
  final Map<int, CategoricalDataEncoderType> _indexToEncoderType;
  final Map<String, List<Object>> _categories;
  final CategoricalDataEncoderType _fallbackEncoderType;

  static const String _loggerPrefix = 'MLData';

  List<List<dynamic>> _data; // the whole dataset including header
  Matrix _features;
  Matrix _labels;
  List<String> _header;
  DataFrameHeaderExtractor _headerExtractor;
  DataFrameFeaturesExtractor _featuresExtractor;
  DataFrameLabelsExtractor _labelsExtractor;
  Map<int, CategoricalDataEncoder> _encoders;

  @override
  Future<List<String>> get header async {
    _data ??= (await _prepareData(_rows, _columns));
    _header ??= _headerExists ? _headerExtractor.extract(_data) : null;
    return _header;
  }

  @override
  Future<Matrix> get features async {
    _data ??= (await _prepareData(_rows, _columns));
    _features ??=
        Matrix.from(_featuresExtractor.getFeatures(), dtype: _dtype);
    return _features;
  }

  @override
  Future<Matrix> get labels async {
    _data ??= (await _prepareData(_rows, _columns));
    _labels ??= Matrix.from(_labelsExtractor.getLabels(), dtype: _dtype);
    return _labels;
  }

  Future<List<List<dynamic>>> _prepareData(
      [Iterable<Tuple2<int, int>> rows,
      Iterable<Tuple2<int, int>> columns]) async {
    final fileStream = _file.openRead();
    final data = await (fileStream
        .transform(utf8.decoder)
        .transform(_csvCodec.decoder)
        .toList());
    final rowsNum = data.length;
    final columnsNum = data.first.length;
    final readMaskCreator = _readMaskCreatorFactory.create(_logger);
    final rowsMask = readMaskCreator.create(
        rows ?? [Tuple2<int, int>(0, rowsNum - (_headerExists ? 2 : 1))]);
    final columnsMask = readMaskCreator
        .create(columns ?? [Tuple2<int, int>(0, columnsNum - 1)]);
    final records = data.sublist(_headerExists ? 1 : 0);

    if (_labelIdx >= records.first.length || _labelIdx < 0) {
      throw RangeError.range(_labelIdx, 0, records.first.length - 1, null,
          _wrapErrorMessage('Invalid label column number'));
    }

    final originalHeader = _headerExists
        ? data[0].map((dynamic el) => el.toString()).toList(growable: true)
        : <String>[];
    final encodersProcessor = _encodersProcessorFactory.create(records,
        originalHeader, _encoderFactory, _fallbackEncoderType, _logger);
    _encoders = encodersProcessor.createEncoders(
        _indexToEncoderType, _nameToEncoderType, _categories);

    _headerExtractor = _headerExtractorFactory.create(columnsMask);
    _featuresExtractor = _featuresExtractorFactory.create(records, rowsMask,
        columnsMask, _encoders, _labelIdx, _valueConverter, _logger);
    _labelsExtractor = _labelsExtractorFactory.create(
        records, rowsMask, _labelIdx, _valueConverter, _encoders, _logger);

    return data;
  }

  String _wrapErrorMessage(String text) => '$_loggerPrefix: $text';

  @override
  Iterable<String> decode(Matrix column, {String colName, int colIdx}) {
    if (colName == null && colIdx == null) {
      throw Exception('Neither column name, nor column index are provided');
    }
    if (colName != null) {
      if (!_headerExists) {
        throw Exception('Column name `$colName` provided, but the data frame '
            'does not have a header');
      }
      if (!_header.contains(colName)) {
        throw Exception('Provided column name `$colName` is not in the header. '
            'Maybe provided column was cutted out during data preparation?');
      }
    }

    if (colIdx != null && (colIdx < 0 || colIdx >= _header.length)) {
      throw RangeError.index(colIdx, _header, 'Wrong column index is provided');
    }

    final idx = colIdx != null ? colIdx : _header.indexOf(colName);
    if (!_encoders.containsKey(idx)) {
      throw Exception('Provided column is not a categorical column');
    }
    throw UnimplementedError('`Decode` is unimplemented yet');
  }
}
