import 'package:ml_algo/src/data_preprocessing/ml_data/labels_extractor/labels_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/labels_extractor/labels_extractor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/labels_extractor/labels_extractor_impl.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/value_converter/value_converter.dart';

class MLDataLabelsExtractorFactoryImpl implements MLDataLabelsExtractorFactory {
  const MLDataLabelsExtractorFactoryImpl();

  @override
  MLDataLabelsExtractor create(List<bool> readMask, int labelIdx, MLDataValueConverter valueConverter) =>
      MLDataLabelsExtractorImpl(readMask, labelIdx, valueConverter);
}
