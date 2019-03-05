import 'package:logging/logging.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/features_extractor/features_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/features_extractor/features_extractor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/features_extractor/features_extractor_impl.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/value_converter/value_converter.dart';

class MLDataFeaturesExtractorFactoryImpl
    implements MLDataFeaturesExtractorFactory {
  const MLDataFeaturesExtractorFactoryImpl();

  @override
  MLDataFeaturesExtractor create(
          List<List<Object>> records,
          List<bool> rowMask,
          List<bool> columnsMask,
          Map<int, CategoricalDataEncoder> encoders,
          int labelIdx,
          MLDataValueConverter valueConverter,
          Logger logger) =>
      MLDataFeaturesExtractorImpl(records, rowMask, columnsMask, encoders,
          labelIdx, valueConverter, logger);
}
