import 'package:ml_dataframe/ml_dataframe.dart';

abstract class FeatureTargetSplitter {
  Iterable<DataFrame> split(DataFrame dataset, {
    Iterable<int> targetIndices = const [],
    Iterable<String> targetNames = const [],
  });
}
