import 'package:logging/logging.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/labels_extractor/labels_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/labels_extractor/labels_extractor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/labels_extractor/labels_extractor_impl.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/value_converter/value_converter.dart';

class MLDataLabelsExtractorFactoryImpl implements MLDataLabelsExtractorFactory {
  const MLDataLabelsExtractorFactoryImpl();

  @override
  MLDataLabelsExtractor create(List<List<Object>> records, List<bool> readMask,
          int labelIdx, MLDataValueConverter valueConverter,
      Map<int, CategoricalDataEncoder> encoders, Logger logger) =>
      MLDataLabelsExtractorImpl(records, readMask, labelIdx, valueConverter,
          encoders, logger);
}
