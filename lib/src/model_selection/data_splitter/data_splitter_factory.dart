import 'package:ml_algo/src/model_selection/data_splitter/data_splitter.dart';
import 'package:ml_algo/src/model_selection/data_splitter/data_splitter_type.dart';

abstract class DataSplitterFactory {
  DataSplitter createByType(DataSplitterType splitterType, {
    int numberOfFolds,
    int p,
  });
}
