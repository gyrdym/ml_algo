import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/to_float_number_converter/to_float_number_converter.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/variables_extractor/variables_extractor.dart';

abstract class VariablesExtractorFactory {
  VariablesExtractor create(
      List<List<Object>> records,
      List<bool> rowMask,
      List<bool> columnsMask,
      Map<int, CategoricalDataEncoder> encoders,
      int labelIdx,
      ToFloatNumberConverter valueConverter,
      Type dtype);
}
