import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_mixin.dart';
import 'package:ml_linalg/vector.dart';

class OrdinalEncoder with EncoderMixin implements CategoricalDataEncoder {
  @override
  Vector encodeLabel(String value, Iterable<String> categoryLabels) {
    final ordinalNum = categoryLabels.toList(growable: false).indexOf(value)
        .toDouble();
    return Vector.from([ordinalNum]);
  }
}
