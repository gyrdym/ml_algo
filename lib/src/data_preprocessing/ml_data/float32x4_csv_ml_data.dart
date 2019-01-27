import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:csv/csv.dart';
import 'package:logging/logging.dart';
import 'package:ml_algo/float32x4_csv_ml_data.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encode_unknown_strategy_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/encoders_processor/encoders_processor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/encoders_processor/encoders_processor_factory_impl.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/features_extractor/features_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/features_extractor/features_extractor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/features_extractor/features_extractor_factory_impl.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/header_extractor/header_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/header_extractor/header_extractor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/header_extractor/header_extractor_factory_impl.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/labels_extractor/labels_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/labels_extractor/labels_extractor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/labels_extractor/labels_extractor_factory_impl.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/read_mask_creator/read_mask_creator_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/read_mask_creator/read_mask_creator_factory_impl.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/validator/ml_data_params_validator.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/validator/ml_data_params_validator_impl.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/value_converter/value_converter.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/value_converter/value_converter_impl.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:tuple/tuple.dart';

class Float32x4CsvMLDataInternal implements Float32x4CsvMLData {
  final CsvCodec _csvCodec;
  final File _file;
  final int _labelIdx;
  final bool _headerExists;
  final List<Tuple2<int, int>> _rows;
  final List<Tuple2<int, int>> _columns;
  final CategoricalDataEncoderFactory _encoderFactory;
  final MLDataParamsValidator _paramsValidator;
  final MLDataValueConverter _valueConverter;
  final MLDataReadMaskCreatorFactory _readMaskCreatorFactory;
  final MLDataHeaderExtractorFactory _headerExtractorFactory;
  final MLDataFeaturesExtractorFactory _featuresExtractorFactory;
  final MLDataLabelsExtractorFactory _labelsExtractorFactory;
  final MLDataEncodersProcessorFactory _encodersProcessorFactory;
  final Logger _logger;

  final Map<String, CategoricalDataEncoderType> _nameToEncoderType;
  final Map<int, CategoricalDataEncoderType> _indexToEncoderType;
  final Map<String, List<Object>> _categories;
  final CategoricalDataEncoderType _fallbackEncoderType;

  static const String _loggerPrefix = 'Float32x4CsvMLData';

  List<List<dynamic>> _data; // the whole dataset including header
  MLMatrix<Float32x4> _features;
  MLVector<Float32x4> _labels;
  List<String> _header;
  MLDataHeaderExtractor _headerExtractor;
  MLDataFeaturesExtractor _featuresExtractor;
  MLDataLabelsExtractor _labelsExtractor;

  Float32x4CsvMLDataInternal.fromFile(String fileName, {
    // public parameters
    String eol = '\n',
    int labelIdx,
    bool headerExists = true,
    CategoricalDataEncoderType encoderType = CategoricalDataEncoderType.oneHot,
    EncodeUnknownValueStrategy encodeUnknownStrategy = EncodeUnknownValueStrategy.throwError,
    Map<String, List<Object>> categories,
    Map<int, List<Object>> categoriesByIndexes,
    Map<String, CategoricalDataEncoderType> categoryNameToEncoder,
    Map<int, CategoricalDataEncoderType> categoryIndexToEncoder,
    List<Tuple2<int, int>> rows,
    List<Tuple2<int, int>> columns,

    // private parameters, they are hidden by the factory
    CategoricalDataEncoderFactory encoderFactory = const CategoricalDataEncoderFactory(),
    MLDataParamsValidator paramsValidator = const MLDataParamsValidatorImpl(),
    MLDataValueConverter valueConverter = const MLDataValueConverterImpl(),
    MLDataHeaderExtractorFactory headerExtractorFactory = const MLDataHeaderExtractorFactoryImpl(),
    MLDataFeaturesExtractorFactory featuresExtractorFactory = const MLDataFeaturesExtractorFactoryImpl(),
    MLDataLabelsExtractorFactory labelsExtractorFactory = const MLDataLabelsExtractorFactoryImpl(),
    MLDataReadMaskCreatorFactory readMaskCreatorFactory = const MLDataReadMaskCreatorFactoryImpl(),
    MLDataEncodersProcessorFactory encodersProcessorFactory = const MLDataEncodersProcessorFactoryImpl(),
    Logger logger,
  }) :
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

  @override
  Future<List<String>> get header async {
    _data ??= (await _prepareData(_rows, _columns));
    _header ??= _headerExists ? _headerExtractor.extract(_data) : null;
    return _header;
  }

  @override
  Future<MLMatrix<Float32x4>> get features async {
    _data ??= (await _prepareData(_rows, _columns));
    _features ??= MLMatrix<Float32x4>.from(_featuresExtractor.getFeatures());
    return _features;
  }

  @override
  Future<MLVector<Float32x4>> get labels async {
    _data ??= (await _prepareData(_rows, _columns));
    _labels ??= MLVector<Float32x4>.from(_labelsExtractor.getLabels());
    return _labels;
  }

  Future<List<List<dynamic>>> _prepareData([Iterable<Tuple2<int, int>> rows, Iterable<Tuple2<int, int>> columns]) async {
    final fileStream = _file.openRead();
    final data = await (fileStream.transform(utf8.decoder).transform(_csvCodec.decoder).toList());
    final rowsNum = data.length;
    final columnsNum = data.first.length;
    final readMaskCreator = _readMaskCreatorFactory.create(_logger);
    final rowsMask = readMaskCreator.create(rows ?? [Tuple2<int, int>(0, rowsNum - (_headerExists ? 2 : 1))]);
    final columnsMask = readMaskCreator.create(columns ?? [Tuple2<int, int>(0, columnsNum - 1)]);
    final records = data.sublist(_headerExists ? 1 : 0);

    if (_labelIdx >= records.first.length || _labelIdx < 0) {
      throw RangeError.range(_labelIdx, 0, records.first.length - 1, null,
          _wrapErrorMessage('Invalid label column number'));
    }

    final originalHeader = _headerExists
        ? data[0].map((dynamic el) => el.toString()).toList(growable: true)
        : <String>[];
    final encodersProcessor = _encodersProcessorFactory.create(records, originalHeader, _encoderFactory,
        _fallbackEncoderType, _logger);
    final encoders = encodersProcessor.createEncoders(_indexToEncoderType, _nameToEncoderType, _categories);

    _headerExtractor = _headerExtractorFactory.create(columnsMask);
    _featuresExtractor = _featuresExtractorFactory.create(records, rowsMask, columnsMask, encoders, _labelIdx,
        _valueConverter, _logger);
    _labelsExtractor = _labelsExtractorFactory.create(records, rowsMask, _labelIdx, _valueConverter, _logger);

    return data;
  }

  String _wrapErrorMessage(String text) => '$_loggerPrefix: $text';
}
