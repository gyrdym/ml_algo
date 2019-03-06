import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/validator/error_messages.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/validator/params_validator.dart';
import 'package:tuple/tuple.dart';

class DataFrameParamsValidatorImpl implements DataFrameParamsValidator {
  const DataFrameParamsValidatorImpl();

  @override
  String validate({
    int labelIdx,
    String labelName,
    Iterable<Tuple2<int, int>> rows,
    Iterable<Tuple2<int, int>> columns,
    bool headerExists = true,
    Map<String, Iterable<Object>> predefinedCategories,
    Map<String, CategoricalDataEncoderType> namesToEncoders,
    Map<int, CategoricalDataEncoderType> indexToEncoder,
  }) {
    final validators = [
      () => _validateHeaderExistsParam(headerExists),
      () => _validatePredefinedCategories(predefinedCategories, headerExists),
      () => _validateNamesToEncoders(namesToEncoders, headerExists),
      () => _validateLabelPosition(labelIdx, labelName, headerExists),
      () => _validateRanges(rows),
      () => _validateRanges(columns, labelIdx),
    ];
    for (int i = 0; i < validators.length; i++) {
      final errorMsg = validators[i]();
      if (errorMsg != '') {
        return errorMsg;
      }
    }
    return '';
  }

  String _validateHeaderExistsParam(bool headerExists) {
    if (headerExists == null) {
      return DataFrameParametersValidationErrorMessages
          .noHeaderExistsParameterProvidedMsg();
    }
    return DataFrameParametersValidationErrorMessages.noErrorMsg;
  }

  String _validatePredefinedCategories(
      Map<String, Iterable<Object>> categories, bool headerExists) {
    if (categories?.isEmpty == true) {
      return DataFrameParametersValidationErrorMessages.emptyCategoriesMsg();
    }
    if (categories != null && !headerExists) {
      return DataFrameParametersValidationErrorMessages
          .noHeaderProvidedMsg(categories);
    }
    return DataFrameParametersValidationErrorMessages.noErrorMsg;
  }

  String _validateNamesToEncoders(
      Map<String, CategoricalDataEncoderType> namesToEncoders,
      bool headerExists) {
    if (namesToEncoders?.isEmpty == true) {
      return DataFrameParametersValidationErrorMessages.emptyEncodersMsg();
    }
    if (!headerExists && namesToEncoders?.isNotEmpty == true) {
      return DataFrameParametersValidationErrorMessages
          .noHeaderProvidedForColumnEncodersMsg(namesToEncoders);
    }
    return DataFrameParametersValidationErrorMessages.noErrorMsg;
  }

  String _validateLabelPosition(int labelIdx, String labelName,
      bool headerExists) {
    if (labelIdx == null && labelName == null) {
      return DataFrameParametersValidationErrorMessages.noLabelPositionMsg();
    }
    if (labelName != null && headerExists == false) {
      return DataFrameParametersValidationErrorMessages
          .labelNameWithoutHeader();
    }
    return DataFrameParametersValidationErrorMessages.noErrorMsg;
  }

  String _validateRanges(Iterable<Tuple2<int, int>> ranges, [int labelIdx]) {
    if (ranges == null || ranges.isEmpty == true) {
      return DataFrameParametersValidationErrorMessages.noErrorMsg;
    }
    Tuple2<int, int> prevRange;
    bool isLabelInRanges = false;

    for (final range in ranges) {
      if (range.item1 > range.item2) {
        return DataFrameParametersValidationErrorMessages
            .leftBoundGreaterThanRightMsg(range);
      }
      if (prevRange != null && prevRange.item2 >= range.item1) {
        return DataFrameParametersValidationErrorMessages
            .intersectingRangesMsg(prevRange, range);
      }
      if (labelIdx != null &&
          labelIdx >= range.item1 &&
          labelIdx <= range.item2) {
        isLabelInRanges = true;
      }
      prevRange = range;
    }

    if (labelIdx != null && !isLabelInRanges) {
      return DataFrameParametersValidationErrorMessages
          .labelIsNotInRangesMsg(labelIdx, ranges);
    }
    return DataFrameParametersValidationErrorMessages.noErrorMsg;
  }
}
