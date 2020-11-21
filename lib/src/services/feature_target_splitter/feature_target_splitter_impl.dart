import 'package:ml_algo/src/services/feature_target_splitter/feature_target_splitter.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:quiver/iterables.dart';

class FeatureTargetSplitterImpl implements
    FeatureTargetSplitter {

  const FeatureTargetSplitterImpl();

  @override
  Iterable<DataFrame> split(DataFrame dataset, {
    Iterable<int> targetIndices = const [],
    Iterable<String> targetNames = const [],
  }) {
    if (targetIndices.isNotEmpty) {
      final uniqueTargetIndices = Set<int>.from(targetIndices);

      final featuresIndices = enumerate(dataset.header)
          .where((indexedName) => !uniqueTargetIndices.contains(indexedName.index))
          .map((indexedName) => indexedName.index);

      return [
        dataset.sampleFromSeries(indices: featuresIndices),
        dataset.sampleFromSeries(indices: uniqueTargetIndices),
      ];
    }

    if (targetNames.isNotEmpty) {
      final uniqueTargetNames = Set<String>.from(targetNames);

      final featuresNames = dataset
          .header
          .where((name) => !uniqueTargetNames.contains(name));

      return [
        dataset.sampleFromSeries(names: featuresNames),
        dataset.sampleFromSeries(names: uniqueTargetNames),
      ];
    }

    throw Exception('Neither target index, nor target name are provided');
  }

}
