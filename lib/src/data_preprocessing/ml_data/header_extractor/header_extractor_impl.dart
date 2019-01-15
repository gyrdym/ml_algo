import 'package:ml_algo/src/data_preprocessing/ml_data/header_extractor/header_extractor.dart';

class MLDataHeaderExtractorImpl implements MLDataHeaderExtractor {
  const MLDataHeaderExtractorImpl();

  @override
  List<String> extract(List<List> data, int columnsNum, [List<bool> readMask]) {
    final headerRow = data[0];
    final header = List<String>(columnsNum);
    int _i = 0;
    for (int i = 0; i < headerRow.length; i++) {
      if (readMask == null || readMask[i] == true) {
        header[_i] = headerRow[i].toString();
        _i++;
      }
    }
    return header;
  }

}