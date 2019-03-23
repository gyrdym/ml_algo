import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/value_converter/value_converter.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/variables_extractor/variables_extractor.dart';

abstract class VariablesExtractorFactory {
  VariablesExtractor create(
      List<List<Object>> records,
      List<bool> rowMask,
      List<bool> columnsMask,
      Map<int, CategoricalDataEncoder> encoders,
      int labelIdx,
      DataFrameValueConverter valueConverter,
      Type dtype);
}
