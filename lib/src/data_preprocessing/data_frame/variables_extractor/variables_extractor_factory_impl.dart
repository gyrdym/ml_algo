import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/to_float_number_converter/to_float_number_converter.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/variables_extractor/variables_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/variables_extractor/variables_extractor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/variables_extractor/variables_extractor_impl.dart';

class VariablesExtractorFactoryImpl implements VariablesExtractorFactory {
  const VariablesExtractorFactoryImpl();

  @override
  VariablesExtractor create(
          List<List<Object>> records,
          List<bool> rowMask,
          List<bool> columnsMask,
          Map<int, CategoricalDataEncoder> encoders,
          int labelIdx,
          ToFloatNumberConverter valueConverter,
          Type dtype) =>
      VariablesExtractorImpl(records, rowMask, columnsMask, encoders,
          labelIdx, valueConverter, dtype);
}
