import 'package:tuple/tuple.dart';

abstract class MLDataReadMaskCreator {
  List<bool> create(Iterable<Tuple2<int, int>> ranges);
}
