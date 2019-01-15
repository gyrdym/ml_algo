import 'dart:math' as math;

import 'package:ml_algo/src/data_preprocessing/ml_data/read_mask_creator/read_mask_creator.dart';
import 'package:tuple/tuple.dart';

class MLDataReadMaskCreatorImpl implements MLDataReadMaskCreator {
  const MLDataReadMaskCreatorImpl();

  @override
  List<bool> create(Iterable<Tuple2<int, int>> ranges, int limit) {
    final mask = List<bool>.filled(limit, false);
    ranges.take(limit).forEach((Tuple2<int, int> range) {
      if (range.item1 >= limit) {
        return false;
      }
      final end = math.min(limit, range.item2 + 1);
      mask.fillRange(range.item1, end, true);
    });
    return mask;
  }
}
