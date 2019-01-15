import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/features_extractor/features_extractor.dart';

abstract class MLDataFeaturesExtractorFactory {
  MLDataFeaturesExtractor create(List<bool> rowMask, List<bool> columnsMask,
      Map<int, CategoricalDataEncoder> indexToEncoder, int labelIdx);
}
