import 'dart:math' as math;

import 'package:ml_algo/src/data_preprocessing/data_frame/read_mask_creator/read_mask_creator.dart';
import 'package:tuple/tuple.dart';

class DataFrameReadMaskCreatorImpl implements DataFrameReadMaskCreator {

  DataFrameReadMaskCreatorImpl();

  static const String emptyRangesMsg =
      'Columns/rows read ranges list cannot be empty!';

  @override
  List<bool> create(Iterable<Tuple2<int, int>> ranges) {
    if (ranges.isEmpty) {
      throw Exception(emptyRangesMsg);
    }
    final numberOfElements = ranges.last.item2 + 1;
    final mask = List<bool>.filled(numberOfElements, false);
    ranges.forEach((Tuple2<int, int> range) {
      if (range.item1 >= numberOfElements) {
        return false;
      }
      final end = math.min(numberOfElements, range.item2 + 1);
      mask.fillRange(range.item1, end, true);
    });
    return mask;
  }
}
