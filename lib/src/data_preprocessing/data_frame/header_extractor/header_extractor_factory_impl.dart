import 'package:ml_algo/src/data_preprocessing/data_frame/header_extractor/header_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/header_extractor/header_extractor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/header_extractor/header_extractor_impl.dart';

class DataFrameHeaderExtractorFactoryImpl implements
    DataFrameHeaderExtractorFactory {

  const DataFrameHeaderExtractorFactoryImpl();

  @override
  DataFrameHeaderExtractor create(List<bool> readMask) =>
      DataFrameHeaderExtractorImpl(readMask);
}
