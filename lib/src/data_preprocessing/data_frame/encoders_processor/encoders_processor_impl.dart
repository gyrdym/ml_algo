import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/encoders_processor/encoders_processor.dart';

class EncodersProcessorImpl implements EncodersProcessor {
  EncodersProcessorImpl(this.records, this.header,
      this.encoderFactory);

  final List<List<Object>> records;
  final List<String> header;
  final CategoricalDataEncoderFactory encoderFactory;

  @override
  Map<int, CategoricalDataEncoder> createEncoders(
      Map<int, CategoricalDataEncoderType> indexesToEncoderTypes,
      Map<String, CategoricalDataEncoderType> namesToEncoderTypes,
  ) {
    if (indexesToEncoderTypes.isNotEmpty) {
      return _createEncodersFromIndexToEncoder(indexesToEncoderTypes);
    } else if (header.isNotEmpty && namesToEncoderTypes.isNotEmpty) {
      return _createEncodersFromNameToEncoder(namesToEncoderTypes);
    }
    return {};
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
