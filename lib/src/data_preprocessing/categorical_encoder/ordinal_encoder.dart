import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_mixin.dart';
import 'package:ml_algo/src/default_parameter_values.dart';
import 'package:ml_linalg/vector.dart';

class OrdinalEncoder with EncoderMixin implements CategoricalDataEncoder {
  OrdinalEncoder([Type dtype = DefaultParameterValues.dtype]) : dtype = dtype;

  @override
  final Type dtype;

  @override
  Vector encodeLabel(String value, Iterable<String> categoryLabels) {
    final ordinalNum = categoryLabels.toList(growable: false).indexOf(value)
        .toDouble();
    return Vector.from([ordinalNum], dtype: dtype);
  }
}
