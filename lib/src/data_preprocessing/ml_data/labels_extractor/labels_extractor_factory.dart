import 'package:logging/logging.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/labels_extractor/labels_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/value_converter/value_converter.dart';

abstract class MLDataLabelsExtractorFactory {
  MLDataLabelsExtractor create(List<List<Object>> records, List<bool> readMask,
      int labelIdx, MLDataValueConverter valueConverter,
      Map<int, CategoricalDataEncoder> encoders, Logger logger);
}
