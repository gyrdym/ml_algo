import 'package:dart_ml/src/data_splitters/splitter_helper.dart';
import 'package:dart_ml/src/data_splitters/splitter_interface.dart';

class KFoldSplitter implements SplitterInterface {
  final SplitterHelper _helper = const SplitterHelper();

  @override
  List<List<int>> split(int numberOfSamples, {int numberOfFolds = 5}) =>
      _helper.split(numberOfSamples, numberOfFolds: numberOfFolds);
}
