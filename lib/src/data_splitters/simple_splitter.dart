import 'package:dart_ml/src/data_splitters/splitter_interface.dart';

class SimpleDataSplitter implements DataSplitterInterface {
  double ratio = .6;

  SimpleDataSplitter({this.ratio});

  List<List<int>> split(int samples) {
    int firstPartLength = (samples * ratio).round();

    return [
      new List.generate(firstPartLength, (int index) => index)
        ..shuffle(),
      new List.generate(samples - firstPartLength, (int index) => firstPartLength + index)
        ..shuffle()
    ];
  }
}