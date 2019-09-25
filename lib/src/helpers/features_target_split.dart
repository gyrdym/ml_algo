import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:quiver/iterables.dart';

Iterable<DataFrame> featuresTargetSplit(DataFrame dataset, {
  Iterable<int> targetIndices = const [],
  Iterable<String> targetNames = const [],
}) {
  if (targetIndices.isNotEmpty != null) {
    final uniqueTargetIndices = Set<int>.from(targetIndices);

    final featuresIndices = enumerate(dataset.header)
        .where((indexedName) => !uniqueTargetIndices.contains(indexedName.index))
        .map((indexedName) => indexedName.index);

    return dataset
        .sampleFromSeries(indices: [featuresIndices, uniqueTargetIndices]);
  }

  if (targetNames.isNotEmpty) {
    final uniqueTargetNames = Set<String>.from(targetNames);

     final featuresNames = dataset
         .header
         .where((name) => !uniqueTargetNames.contains(name));

     return dataset
         .sampleFromSeries(names: [featuresNames, uniqueTargetNames]);
  }

  throw Exception('Neither target index, nor target name are provided');
}