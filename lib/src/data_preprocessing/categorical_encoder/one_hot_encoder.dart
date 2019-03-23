import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_mixin.dart';
import 'package:ml_algo/src/utils/default_parameter_values.dart';
import 'package:ml_linalg/vector.dart';

class OneHotEncoder with EncoderMixin implements CategoricalDataEncoder {
  OneHotEncoder([Type dtype = DefaultParameterValues.dtype]) : dtype = dtype;

  @override
  final Type dtype;

  @override
  Vector encodeLabel(String value, Iterable<String> categoryLabels) {
    final valueIdx = categoryLabels.toList(growable: false).indexOf(value);
    final encodedCategorySource = List<double>.generate(
        categoryLabels.length,
            (int idx) => idx == valueIdx ? 1.0 : 0.0
    );
    return Vector.from(encodedCategorySource, dtype: dtype);
  }
}
