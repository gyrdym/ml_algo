abstract class MLDataHeaderExtractor {
  List<String> extract(List<List<dynamic>> data, int columnsNum, [List<bool> readMask]);
}