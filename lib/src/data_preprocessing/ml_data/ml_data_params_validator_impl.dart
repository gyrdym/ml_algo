import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/ml_data_params_validator.dart';
import 'package:tuple/tuple.dart';

class MLDataParamsValidatorImpl implements MLDataParamsValidator {
  const MLDataParamsValidatorImpl();

  static const noErrorMsg = '';

  static String labelIndexMustNotBeNullMsg() =>
      'label index must not be null';

  static String leftBoundGreaterThanRightMsg(Tuple2<int, int> range) =>
      'left boundary of the range $range is greater than the right one';

  static String intersectingRangesMsg(Tuple2<int, int> range1, Tuple2<int, int> range2) =>
      '$range1 and $range2 ranges are intersecting';

  static String labelIsNotInRanges(int labelIdx, Iterable<Tuple2<int, int>> ranges) =>
      'label index $labelIdx is not in provided ranges $ranges';

  @override
  String validate({
    int labelIdx,
    Iterable<Tuple2<int, int>> rows,
    Iterable<Tuple2<int, int>> columns,
    bool headerExists,
    Map<String, Iterable<Object>> predefinedCategories,
    Map<String, CategoricalDataEncoderType> nameToEncoder,
    Map<int, CategoricalDataEncoderType> indexToEncoder
  }) {
    final validators = [
          () => _validateLabelIdx(labelIdx),
          () => _validateReadRanges(rows),
          () => _validateReadRanges(columns, labelIdx),
    ];
    for (int i = 0; i < validators.length; i++) {
      final errorMsg = validators[i]();
      if (errorMsg != '') {
        return errorMsg;
      }
    }
    return '';
  }

  String _validateLabelIdx(int labelIdx) {
    if (labelIdx == null) {
      return labelIndexMustNotBeNullMsg();
    }
    return noErrorMsg;
  }

  String _validateReadRanges(Iterable<Tuple2<int, int>> ranges, [int labelIdx]) {
    if (ranges == null) {
      return '';
    }

    String errorMessage = '';
    Tuple2<int, int> prevRange;
    bool isLabelInRanges = false;

    ranges.forEach((Tuple2<int, int> range) {
      if (range.item1 > range.item2) {
        errorMessage = leftBoundGreaterThanRightMsg(range);
      }
      if (prevRange != null && prevRange.item2 >= range.item1) {
        errorMessage = intersectingRangesMsg(prevRange, range);
      }
      if (labelIdx != null && labelIdx >= range.item1 && labelIdx <= range.item2) {
        isLabelInRanges = true;
      }
      prevRange = range;
    });

    if (labelIdx != null && !isLabelInRanges) {
      errorMessage = labelIsNotInRanges(labelIdx, ranges);
    }

    return errorMessage;
  }
}