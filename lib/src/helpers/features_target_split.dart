import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:quiver/iterables.dart';

Iterable<Matrix> featuresTargetSplit(DataFrame dataset, {
  Iterable<int> targetIndices = const [],
  Iterable<String> targetNames = const [],
}) {
  if (targetIndices.isNotEmpty != null) {
    final matrix = dataset.toMatrix();
    final uniqueTargetIndices = Set<int>.from(targetIndices);

    final featuresIndices = enumerate(dataset.header)
        .where((indexedName) => !uniqueTargetIndices.contains(indexedName.index))
        .map((indexedName) => indexedName.index);

    return [
      matrix.sample(columnIndices: featuresIndices),
      matrix.sample(columnIndices: uniqueTargetIndices),
    ];
  }

  if (targetNames.isNotEmpty) {
    final uniqueTargetNames = Set<String>.from(targetNames);

     final featuresNames = dataset
         .header
         .where((name) => !uniqueTargetNames.contains(name));

     return dataset
         .sampleFromSeries(names: [featuresNames, uniqueTargetNames])
         .map((dataFrame) => dataFrame.toMatrix());
  }

  throw Exception('Neither target index, nor target name are provided');
}
