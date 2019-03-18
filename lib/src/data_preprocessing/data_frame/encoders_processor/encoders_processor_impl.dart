import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/encoders_processor/encoders_processor.dart';

class DataFrameEncodersProcessorImpl implements DataFrameEncodersProcessor {
  DataFrameEncodersProcessorImpl(this.records, this.header, this.encoderFactory,
      this.fallbackEncoderType);

  final List<List<Object>> records;
  final List<String> header;
  final CategoricalDataEncoderFactory encoderFactory;
  final CategoricalDataEncoderType fallbackEncoderType;

  static const String noHeaderProvidedWarningMsg =
      'Column names with categorical values are provided, but there are no '
      'header with column names!';

  @override
  Map<int, CategoricalDataEncoder> createEncoders(
      Map<int, CategoricalDataEncoderType> indexesToEncoderTypes,
      Map<String, CategoricalDataEncoderType> namesToEncoderTypes,
      Map<String, List<String>> categories) {
    Map<int, CategoricalDataEncoder> encoders = {};
    if (indexesToEncoderTypes.isNotEmpty) {
      encoders = _createEncodersFromIndexToEncoder(indexesToEncoderTypes);
    } else if (header.isNotEmpty) {
      if (namesToEncoderTypes.isNotEmpty) {
        encoders = _createEncodersFromNameToEncoder(namesToEncoderTypes);
      } else if (categories.isNotEmpty) {
        encoders = _createEncodersFromCategories(categories);
      }
    } else if (namesToEncoderTypes.isNotEmpty || categories.isNotEmpty) {
//      logger.warning(noHeaderProvidedWarningMsg);
    }
    return encoders;
  }

  Map<int, CategoricalDataEncoder> _createEncodersFromNameToEncoder(
          Map<String, CategoricalDataEncoderType> nameToEncoder) =>
      _createEncoders((int colIdx) {
        final name = header[colIdx];
        return nameToEncoder.containsKey(name) ? nameToEncoder[name] : null;
      });

  Map<int, CategoricalDataEncoder> _createEncodersFromIndexToEncoder(
          Map<int, CategoricalDataEncoderType> indexToEncoderType) =>
      _createEncoders((int colIdx) => indexToEncoderType.containsKey(colIdx)
          ? indexToEncoderType[colIdx]
          : null);

  Map<int, CategoricalDataEncoder> _createEncodersFromCategories(
      Map<String, List<String>> categories) {
    final indexToEncoder = <int, CategoricalDataEncoder>{};
    for (int i = 0; i < header.length; i++) {
      final name = header[i];
      if (categories.containsKey(name)) {
        indexToEncoder[i] = encoderFactory.fromType(fallbackEncoderType);
      }
    }
    return indexToEncoder;
  }

  Map<int, CategoricalDataEncoder> _createEncoders(
      CategoricalDataEncoderType getEncoderType(int colIdx)) {
    final indexToEncoder = <int, CategoricalDataEncoder>{};
    final encodersValues = <int, List<String>>{};
    for (int rowIdx = 0; rowIdx < records.length; rowIdx++) {
      for (int colIdx = 0; colIdx < records[rowIdx].length; colIdx++) {
        final encoderType = getEncoderType(colIdx);
        if (encoderType != null) {
          indexToEncoder.putIfAbsent(
              colIdx, () => encoderFactory.fromType(encoderType));
          encodersValues.putIfAbsent(
              colIdx, () => List<String>(records.length));
          encodersValues[colIdx][rowIdx] = records[rowIdx][colIdx].toString();
        }
      }
    }
    return indexToEncoder;
  }
}
