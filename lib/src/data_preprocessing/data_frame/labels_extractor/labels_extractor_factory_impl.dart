import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/labels_extractor/labels_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/labels_extractor/labels_extractor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/labels_extractor/labels_extractor_impl.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/value_converter/value_converter.dart';

class DataFrameLabelsExtractorFactoryImpl implements
    DataFrameLabelsExtractorFactory {

  const DataFrameLabelsExtractorFactoryImpl();

  @override
  DataFrameLabelsExtractor create(List<List<Object>> records,
      List<bool> readMask, int labelIdx, DataFrameValueConverter valueConverter,
      Map<int, CategoricalDataEncoder> encoders) =>
      DataFrameLabelsExtractorImpl(records, readMask, labelIdx, valueConverter,
          encoders);
}
