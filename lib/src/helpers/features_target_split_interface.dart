import 'package:ml_dataframe/ml_dataframe.dart';

typedef FeaturesTargetSplit = Iterable<DataFrame> Function(DataFrame dataset, {
  Iterable<int> targetIndices,
  Iterable<String> targetNames,
});
