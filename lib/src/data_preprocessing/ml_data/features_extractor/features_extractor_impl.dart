import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/features_extractor/features_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/value_converter/value_converter.dart';

class MLDataFeaturesExtractorImpl implements MLDataFeaturesExtractor {
  final List<bool> rowsMask;
  final List<bool> columnsMask;
  final Map<int, CategoricalDataEncoder> encoders;
  final int rowsNum;
  final int columnsNum;
  final int labelIdx;
  final MLDataValueConverter valueConverter;

  MLDataFeaturesExtractorImpl(this.rowsMask, this.columnsMask, this.encoders, this.labelIdx, this.valueConverter) :
      rowsNum = rowsMask.where((bool flag) => flag).length,
      columnsNum = columnsMask.where((bool flag) => flag).length;

  @override
  List<List<double>> extract(List<List> records, {bool hasCategoricalData = false}) {
    final features = List<List<double>>(rowsNum);
    int _i = 0;
    for (int i = 0; i < records.length; i++) {
      if (rowsMask[i] == true) {
        final featuresRaw = records[i];
        features[_i++] = hasCategoricalData
            ? _convertFeaturesWithCategoricalData(featuresRaw)
            : _convertFeatures(featuresRaw);
      }
    }
    return features;
  }

  /// Light-weight method for data encoding without any checks if the current feature is categorical
  List<double> _convertFeatures(List<Object> features) {
    final converted = List<double>(columnsNum - 1); // minus one column for label values
    int _i = 0;
    for (int i = 0; i < features.length; i++) {
      final feature = features[i];
      if (labelIdx != i && (columnsMask == null || columnsMask[i] == true)) {
        converted[_i++] = valueConverter.convert(feature);
      }
    }
    return converted;
  }

  /// In order to avoid limitless checks if the current feature is categorical, let's create a separate method for
  /// data encoding if we know exactly that categories are presented in the data set
  List<double> _convertFeaturesWithCategoricalData(List<Object> features) {
    final converted = <double>[];
    for (int i = 0; i < features.length; i++) {
      if (labelIdx == i || columnsMask[i] == false) {
        continue;
      }
      final feature = features[i];
      Iterable<double> expanded;
      if (encoders.containsKey(i)) {
        expanded = encoders[i].encode(feature);
      } else {
        expanded = [valueConverter.convert(feature)];
      }
      converted.addAll(expanded);
    }
    return converted;
  }
}
