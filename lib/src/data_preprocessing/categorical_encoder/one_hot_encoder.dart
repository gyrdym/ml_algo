import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_mixin.dart';
import 'package:ml_linalg/vector.dart';

class OneHotEncoder with EncoderMixin implements CategoricalDataEncoder {
  @override
  Vector encodeLabel(String value, Iterable<String> categoryLabels) {
    final valueIdx = categoryLabels.toList(growable: false).indexOf(value);
    final encodedCategorySource = List<double>.generate(
        categoryLabels.length,
            (int idx) => idx == valueIdx ? 1.0 : 0.0
    );
    return Vector.from(encodedCategorySource);
  }
}
