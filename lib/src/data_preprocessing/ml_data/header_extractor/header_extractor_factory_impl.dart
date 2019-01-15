import 'package:ml_algo/src/data_preprocessing/ml_data/header_extractor/header_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/header_extractor/header_extractor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/header_extractor/header_extractor_impl.dart';

class MLDataHeaderExtractorFactoryImpl implements MLDataHeaderExtractorFactory {
  const MLDataHeaderExtractorFactoryImpl();

  @override
  MLDataHeaderExtractor create(List<bool> readMask) => MLDataHeaderExtractorImpl(readMask);
}