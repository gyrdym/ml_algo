import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/encoders_processor/encoders_processor.dart';

class MLDataEncodersProcessorImpl implements MLDataEncodersProcessor {
  final List<String> header;
  final CategoricalDataEncoderFactory encoderFactory;
  final CategoricalDataEncoderType fallbackEncoderType;

  MLDataEncodersProcessorImpl(this.header, this.encoderFactory, this.fallbackEncoderType);

  @override
  Map<int, CategoricalDataEncoder> createEncoders(Map<int, CategoricalDataEncoderType> indexesToEncoderTypes,
      Map<String, CategoricalDataEncoderType> namesToEncoderTypes, Map<String, List<Object>> categories) {

  }

  void _setEncoders(List<List<dynamic>> data) {
    final _isCategoryNameToEncoderTypeDefined =
        _nameToEncoderType != null && _nameToEncoderType.isNotEmpty;

    final _isCategoryIndexToEncoderTypeDefined =
        _indexToEncoderType != null && _indexToEncoderType.isNotEmpty;

    // let's cast all categorical data to column index-based format

    // create _categoryIndexToEncoderType from category names
    if (_headerExists && _isCategoryNameToEncoderTypeDefined && !_isCategoryIndexToEncoderTypeDefined) {
      _fillIndexToEncoderTypeMap();
    }

    // create map "column index" -> "encoder" from fully-predefined categories data - _categories
    if (!_isCategoryIndexToEncoderTypeDefined && _categories?.isNotEmpty == true) {
      _fillIndexToEncoderMapFromCategories();
      // create map "column index" -> "encoder" from map "column index" -> "encoder type"
    } else if (_isCategoryIndexToEncoderTypeDefined) {
      _fillIndexToEncoderMapFromEncoderTypes();
    }

    _setCategoryValues();
  }

  void _fillIndexToEncoderTypeMap() {
    for (int i = 0; i < _originalHeader.length; i++) {
      final name = _originalHeader[i];
      if (_nameToEncoderType.containsKey(name)) {
        _indexToEncoderType.putIfAbsent(i, () => _nameToEncoderType[name]);
      }
    }
  }

  void _fillIndexToEncoderMapFromCategories() {
    for (int i = 0; i < _originalHeader.length; i++) {
      final name = _originalHeader[i];
      if (_categories.containsKey(name)) {
        _indexToEncoder[i] = _encoderFactory.fromType(_fallbackEncoderType);
      }
    }
  }

  void _fillIndexToEncoderMapFromEncoderTypes() {
    for (int i = 0; i < _originalHeader.length; i++) {
      if (_indexToEncoderType.containsKey(i)) {
        final encoderType = _indexToEncoderType[i];
        _indexToEncoder[i] = _encoderFactory.fromType(encoderType);
      }
    }
  }

  void _setCategoryValues() {
    final values = <int, List<Object>>{};
    for (int i = 0; i < _records.length; i++) {
      for (int j = 0; j < _records[i].length; j++) {
        if (_indexToEncoder.containsKey(j)) {
          values.putIfAbsent(j, () => List<Object>(_records.length));
          values[j][i] = _records[i][j];
        }
      }
    }
    values.forEach((int column, List<Object> values) {
      _indexToEncoder[column].setCategoryValues(values);
    });
  }
}
