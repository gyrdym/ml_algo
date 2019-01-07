import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encode_unknown_strategy_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';

class OrdinalEncoder implements CategoricalDataEncoder {
  @override
  final EncodeUnknownValueStrategy encodeUnknownValueStrategy;

  final List<Object> _values;

  OrdinalEncoder([List<Object> values, this.encodeUnknownValueStrategy = EncodeUnknownValueStrategy.throwError]) :
        _values = values;

  @override
  Iterable<double> encode(Object value) {
    if (!_values.contains(value)) {
      if (encodeUnknownValueStrategy == EncodeUnknownValueStrategy.throwError) {
        throw UnsupportedError('Ordinal encoding: unsupported value `$value`');
      } else {
        return [0.0];
      }
    }
    final ordinalNum = _values.indexOf(value).toDouble();
    return [ordinalNum + 1]; // plus one - to avoid zero value. Zero is reserved for unknown values
  }
}
