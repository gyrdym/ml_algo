import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

mixin DataValidationMixin {

  void validateTrainData(DataFrame trainData, Iterable<String> columnNames) {
    columnNames.forEach((name) {
      if (trainData[name] == null) {
        throw Exception('Target column `$name` does not exist in the passed '
            'train data');
      }
    });
  }

  void validateTestFeatures(DataFrame features, DType dtype) {
    if (!features.toMatrix(dtype).hasData) {
      throw Exception('No features provided');
    }
  }
}

