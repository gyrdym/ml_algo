import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:tuple/tuple.dart';

abstract class DataFrameParamsValidator {
  String validate({
    int labelIdx,
    String labelName,
    Iterable<Tuple2<int, int>> rows,
    Iterable<Tuple2<int, int>> columns,
    bool headerExists,
    Map<String, Iterable<Object>> predefinedCategories,
    Map<String, CategoricalDataEncoderType> namesToEncoders,
    Map<int, CategoricalDataEncoderType> indexToEncoder,
  });
}
