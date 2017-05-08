import 'package:dart_ml/src/data_splitters/base_splitter.dart';
import 'package:dart_ml/src/data_splitters/splitter_interface.dart';

class LeaveOneOutSplitter implements SplitterInterface {
  final SplitterHelper _helper = const SplitterHelper();

  @override
  List<List<int>> split(int numberOfSamples) => _helper.split(numberOfSamples, numberOfFolds: numberOfSamples);
}