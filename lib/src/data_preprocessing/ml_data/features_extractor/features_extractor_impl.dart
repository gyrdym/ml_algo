import 'package:logging/logging.dart';
import 'package:ml_algo/src/utils/logger/logger_mixin.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/features_extractor/features_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/value_converter/value_converter.dart';

class MLDataFeaturesExtractorImpl extends Object
    with LoggerMixin
    implements MLDataFeaturesExtractor {

  MLDataFeaturesExtractorImpl(this.records, this.rowsMask, this.columnsMask,
      this.encoders, this.labelIdx, this.valueConverter, this.logger)
      : rowsNum = rowsMask.where((bool flag) => flag).length,
        columnsNum = columnsMask.where((bool flag) => flag).length {
    if (columnsMask.length > records.first.length) {
      throwException(columnsMaskWrongLengthMsg);
    }

    if (rowsMask.length > records.length) {
      throwException(rowsMaskWrongLengthMsg);
    }
  }

  static const String rowsMaskWrongLengthMsg =
      'Rows mask length should not be greater than actual rows number in the dataset!';

  static const String columnsMaskWrongLengthMsg =
      'Columns mask length should not be greater than actual columns number in the dataset!';

  final List<bool> rowsMask;
  final List<bool> columnsMask;
  final Map<int, CategoricalDataEncoder> encoders;
  final int rowsNum;
  final int columnsNum;
  final int labelIdx;
  final MLDataValueConverter valueConverter;
  final List<List<Object>> records;

  @override
  final Logger logger;

  @override
  List<List<double>> getFeatures() {
    final features = List<List<double>>(rowsNum);
    int _i = 0;
    for (int i = 0; i < rowsMask.length; i++) {
      if (rowsMask[i] == true) {
        final featuresRaw = records[i];
        features[_i++] = encoders.isNotEmpty
            ? _convertFeaturesWithCategoricalData(featuresRaw)
            : _convertFeatures(featuresRaw);
      }
    }
    return features;
  }

  /// Light-weight method for data encoding without any checks if the current feature is categorical
  List<double> _convertFeatures(List<Object> features) {
    final converted = <double>[];
    for (int i = 0; i < columnsMask.length; i++) {
      final feature = features[i];
      if (labelIdx != i && columnsMask[i] == true) {
        converted.add(valueConverter.convert(feature));
      }
    }
    return converted;
  }

  /// In order to avoid limitless checks if the current feature is categorical, let's create a separate method for
  /// data encoding if we know exactly that categories are presented in the data set
  List<double> _convertFeaturesWithCategoricalData(List<Object> features) {
    final converted = <double>[];
    for (int i = 0; i < columnsMask.length; i++) {
      if (labelIdx == i || columnsMask[i] == false) {
        continue;
      }
      final feature = features[i];
      Iterable<double> expanded;
      if (encoders.containsKey(i)) {
        expanded = encoders[i].encodeSingle(feature);
      } else {
        expanded = [valueConverter.convert(feature)];
      }
      converted.addAll(expanded);
    }
    return converted;
  }
}
