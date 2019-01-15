import 'package:ml_algo/src/data_preprocessing/ml_data/header_extractor/header_extractor.dart';

class MLDataHeaderExtractorImpl implements MLDataHeaderExtractor {
  final List<bool> readMask;
  final int columnsNum;

  MLDataHeaderExtractorImpl(this.readMask)
      : columnsNum = readMask.where((bool flag) => flag).length;

  @override
  List<String> extract(List<List> data) {
    final headerRow = data[0];
    final header = List<String>(columnsNum);
    int _i = 0;
    for (int i = 0; i < headerRow.length; i++) {
      if (readMask[i] == true) {
        header[_i] = headerRow[i].toString();
        _i++;
      }
    }
    return header;
  }
}