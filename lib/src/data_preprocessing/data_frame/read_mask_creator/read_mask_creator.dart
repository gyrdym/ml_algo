import 'package:tuple/tuple.dart';

abstract class DataFrameReadMaskCreator {
  List<bool> create(Iterable<Tuple2<int, int>> ranges);
}
