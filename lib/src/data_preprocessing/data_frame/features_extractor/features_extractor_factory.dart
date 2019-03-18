import 'package:logging/logging.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/features_extractor/features_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/value_converter/value_converter.dart';

abstract class DataFrameFeaturesExtractorFactory {
  DataFrameFeaturesExtractor create(
      List<List<Object>> records,
      List<bool> rowMask,
      List<bool> columnsMask,
      Map<int, CategoricalDataEncoder> encoders,
      int labelIdx,
      DataFrameValueConverter valueConverter);
}
