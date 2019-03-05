import 'package:logging/logging.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/labels_extractor/labels_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/value_converter/value_converter.dart';

abstract class DataFrameLabelsExtractorFactory {
  DataFrameLabelsExtractor create(List<List<Object>> records, List<bool> readMask,
      int labelIdx, DataFrameValueConverter valueConverter,
      Map<int, CategoricalDataEncoder> encoders, Logger logger);
}
