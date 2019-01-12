import 'package:tuple/tuple.dart';

abstract class MLDataValidationErrorMessages {
  static const noErrorMsg = '';

  static String noHeaderProvided(Map<String, Iterable<Object>> categories) =>
      'no header provided to define, which columns belongs to given categories $categories';

  static String labelIndexMustNotBeNullMsg() =>
      'label index must not be null';

  static String leftBoundGreaterThanRightMsg(Tuple2<int, int> range) =>
      'left boundary of the range $range is greater than the right one';

  static String intersectingRangesMsg(Tuple2<int, int> range1, Tuple2<int, int> range2) =>
      '$range1 and $range2 ranges are intersecting';

  static String labelIsNotInRanges(int labelIdx, Iterable<Tuple2<int, int>> ranges) =>
      'label index $labelIdx is not in provided ranges $ranges';
}
