import 'package:ml_dataframe/ml_dataframe.dart';

void validateTrainData(DataFrame trainData, Iterable<String> columnNames) {
  columnNames.forEach((name) {
    if (trainData[name] == null) {
      throw Exception('Target column `$name` does not exist in the passed '
          'train data');
    }
  });
}