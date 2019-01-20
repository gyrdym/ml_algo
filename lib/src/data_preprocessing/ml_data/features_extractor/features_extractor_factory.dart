import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/features_extractor/features_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/value_converter/value_converter.dart';

abstract class MLDataFeaturesExtractorFactory {
  MLDataFeaturesExtractor create(List<bool> rowMask, List<bool> columnsMask, Map<int, CategoricalDataEncoder> encoders,
      int labelIdx, MLDataValueConverter valueConverter);
}
