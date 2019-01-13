import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/validator/error_messages.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/validator/ml_data_params_validator.dart';
import 'package:tuple/tuple.dart';

class MLDataParamsValidatorImpl implements MLDataParamsValidator {
  const MLDataParamsValidatorImpl();

  @override
  String validate({
    int labelIdx,
    Iterable<Tuple2<int, int>> rows,
    Iterable<Tuple2<int, int>> columns,
    bool headerExists,
    Map<String, Iterable<Object>> predefinedCategories,
    Map<String, CategoricalDataEncoderType> nameToEncoder,
    Map<int, CategoricalDataEncoderType> indexToEncoder,
  }) {
    final validators = [
          () => _validatePredefinedCategories(predefinedCategories, headerExists),
          () => _validateNamesToEncoders(nameToEncoder, headerExists),
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

  String _validatePredefinedCategories(Map<String, Iterable<Object>> categories, bool headerExists) {
    if (!headerExists && categories?.isNotEmpty == true) {
      return MLDataValidationErrorMessages.noHeaderProvided(categories);
    }
    return MLDataValidationErrorMessages.noErrorMsg;
  }

  String _validateNamesToEncoders(Map<String, CategoricalDataEncoderType> namesToEncoders, bool headerExists) {
    if (!headerExists && namesToEncoders?.isEmpty == true) {
      return MLDataValidationErrorMessages.noHeaderProvidedForColumnEncoders(namesToEncoders);
    }
    return MLDataValidationErrorMessages.noErrorMsg;
  }

  String _validateLabelIdx(int labelIdx) {
    if (labelIdx == null) {
      return MLDataValidationErrorMessages.labelIndexMustNotBeNullMsg();
    }
    return MLDataValidationErrorMessages.noErrorMsg;
  }

  String _validateReadRanges(Iterable<Tuple2<int, int>> ranges, [int labelIdx]) {
    if (ranges == null || ranges.isEmpty == true) {
      return MLDataValidationErrorMessages.noErrorMsg;
    }
    Tuple2<int, int> prevRange;
    bool isLabelInRanges = false;

    for (final range in ranges) {
      if (range.item1 > range.item2) {
        return MLDataValidationErrorMessages.leftBoundGreaterThanRightMsg(range);
      }
      if (prevRange != null && prevRange.item2 >= range.item1) {
        return MLDataValidationErrorMessages.intersectingRangesMsg(prevRange, range);
      }
      if (labelIdx != null && labelIdx >= range.item1 && labelIdx <= range.item2) {
        isLabelInRanges = true;
      }
      prevRange = range;
    }

    if (labelIdx != null && !isLabelInRanges) {
      return MLDataValidationErrorMessages.labelIsNotInRanges(labelIdx, ranges);
    }
    return MLDataValidationErrorMessages.noErrorMsg;
  }
}