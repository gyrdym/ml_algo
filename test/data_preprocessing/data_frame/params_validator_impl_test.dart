import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/validator/error_messages.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/validator/params_validator_impl.dart';
import 'package:test/test.dart';
import 'package:tuple/tuple.dart';

void main() {
  group('DataFrameParamsValidatorImpl (`rows` param)', () {
    test('should return no error message if no row ranges are provided', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 0,
        rows: null,
        columns: [const Tuple2<int, int>(0, 1)],
        headerExists: true,
      );
      expect(actual, equals(DataFrameParametersValidationErrorMessages
          .noErrorMsg));
    });

    test('should return no error message if empty row ranges list is '
        'provided', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 0,
        rows: [],
        columns: [const Tuple2<int, int>(0, 1)],
        headerExists: true,
      );
      expect(actual, equals(DataFrameParametersValidationErrorMessages
          .noErrorMsg));
    });

    test('should return proper error message if provided row ranges are '
        'intersecting', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final rowRange1 = const Tuple2<int, int>(0, 2);
      final rowRange2 = const Tuple2<int, int>(0, 1);
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 0,
        rows: [rowRange1, rowRange2],
        columns: [const Tuple2<int, int>(0, 1)],
        headerExists: true,
      );
      expect(
          actual,
          equals(DataFrameParametersValidationErrorMessages
              .intersectingRangesMsg(rowRange1, rowRange2)));
    });

    test('should return proper error message if provided row ranges are '
        'intersecting, corner case', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final rowRange1 = const Tuple2<int, int>(0, 2);
      final rowRange2 = const Tuple2<int, int>(2, 3);
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 0,
        rows: [rowRange1, rowRange2],
        columns: [const Tuple2<int, int>(0, 1)],
        headerExists: true,
      );
      expect(
          actual,
          equals(DataFrameParametersValidationErrorMessages
              .intersectingRangesMsg(rowRange1, rowRange2)));
    });

    test('should return proper error message if at least one of the provided '
        'row ranges has its left bondary greater than the right', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final rowRange1 = const Tuple2<int, int>(0, 2);
      final rowRange2 = const Tuple2<int, int>(5, 3);
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 0,
        rows: [rowRange1, rowRange2],
        columns: [const Tuple2<int, int>(0, 1)],
        headerExists: true,
      );
      expect(
          actual,
          equals(DataFrameParametersValidationErrorMessages
              .leftBoundGreaterThanRightMsg(rowRange2)));
    });

    test('should return no error message if valid row ranges list is '
        'provided', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final rowRange1 = const Tuple2<int, int>(0, 2);
      final rowRange2 = const Tuple2<int, int>(5, 6);
      final rowRange3 = const Tuple2<int, int>(7, 10);
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 0,
        rows: [rowRange1, rowRange2, rowRange3],
        columns: [const Tuple2<int, int>(0, 1)],
        headerExists: true,
      );
      expect(actual, equals(DataFrameParametersValidationErrorMessages
          .noErrorMsg));
    });
  });

  group('DataFrameParamsValidatorImpl (`columns` param)', () {
    test('should return no error message if no row ranges are provided', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 0,
        rows: [const Tuple2<int, int>(0, 1)],
        columns: null,
        headerExists: true,
      );
      expect(actual, equals(DataFrameParametersValidationErrorMessages
          .noErrorMsg));
    });

    test('should return no error message if empty row ranges list is provided',
        () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 0,
        rows: [const Tuple2<int, int>(0, 1)],
        columns: [],
        headerExists: true,
      );
      expect(actual, equals(DataFrameParametersValidationErrorMessages
          .noErrorMsg));
    });

    test('should return proper error message if provided row ranges are '
        'intersecting', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final colRange1 = const Tuple2<int, int>(0, 2);
      final colRange2 = const Tuple2<int, int>(0, 1);
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 0,
        rows: [const Tuple2<int, int>(0, 1)],
        columns: [colRange1, colRange2],
        headerExists: true,
      );
      expect(
          actual,
          equals(DataFrameParametersValidationErrorMessages
              .intersectingRangesMsg(colRange1, colRange2)));
    });

    test('should return proper error message if provided row ranges are '
        'intersecting, corner case', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final colRange1 = const Tuple2<int, int>(0, 2);
      final colRange2 = const Tuple2<int, int>(2, 3);
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 0,
        rows: [const Tuple2<int, int>(0, 1)],
        columns: [colRange1, colRange2],
        headerExists: true,
      );
      expect(
          actual,
          equals(DataFrameParametersValidationErrorMessages
              .intersectingRangesMsg(colRange1, colRange2)));
    });

    test('should return proper error message if at least one of the provided '
        'row ranges has its left bondary greater than the right', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final colRange1 = const Tuple2<int, int>(0, 2);
      final colRange2 = const Tuple2<int, int>(5, 3);
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 0,
        rows: [const Tuple2<int, int>(0, 1)],
        columns: [colRange1, colRange2],
        headerExists: true,
      );
      expect(
          actual,
          equals(DataFrameParametersValidationErrorMessages
              .leftBoundGreaterThanRightMsg(colRange2)));
    });

    test('should return no error message if valid row ranges list is provided',
        () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final colRange1 = const Tuple2<int, int>(0, 2);
      final colRange2 = const Tuple2<int, int>(5, 6);
      final colRange3 = const Tuple2<int, int>(7, 10);
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 0,
        rows: [const Tuple2<int, int>(0, 1)],
        columns: [colRange1, colRange2, colRange3],
        headerExists: true,
      );
      expect(actual,
          equals(DataFrameParametersValidationErrorMessages.noErrorMsg));
    });
  });

  group('DataFrameParamsValidatorImpl (label position parameters)', () {
    test('should return no error message if label index is provided', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 10,
        headerExists: true,
      );
      expect(actual,
          equals(DataFrameParametersValidationErrorMessages.noErrorMsg));
    });

    test('should return proper error message if no label index is provided',
        () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final actual = dataFrameParamsValidator.validate(
        labelIdx: null,
        headerExists: true,
      );
      expect(actual, equals(DataFrameParametersValidationErrorMessages
          .noLabelPositionMsg()));
    });

    test('should return proper error message if provided column ranges list '
        'does not contain label index', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final labelIdx = 10;
      final ranges = [
        const Tuple2<int, int>(0, 5),
        const Tuple2<int, int>(6, 8)
      ];
      final actual = dataFrameParamsValidator.validate(
        labelIdx: labelIdx,
        columns: ranges,
        headerExists: true,
      );
      expect(
          actual,
          equals(DataFrameParametersValidationErrorMessages
              .labelIsNotInRangesMsg(labelIdx, ranges)));
    });

    test('should return proper error message if neither `labelIdx` nor '
        '`labelName` are provided', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final ranges = [
        const Tuple2<int, int>(0, 5),
        const Tuple2<int, int>(6, 8)
      ];
      final actual = dataFrameParamsValidator.validate(
        labelIdx: null,
        labelName: null,
        columns: ranges,
        headerExists: true,
      );
      expect(actual, equals(DataFrameParametersValidationErrorMessages
          .noLabelPositionMsg()));
    });

    test('should return proper error message if `labelName` is provided, '
        'but `headerExists` is false (no header case)', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final ranges = [
        const Tuple2<int, int>(0, 5),
        const Tuple2<int, int>(6, 8)
      ];
      final actual = dataFrameParamsValidator.validate(
        labelIdx: null,
        labelName: 'some name',
        columns: ranges,
        headerExists: false,
      );
      expect(actual, equals(DataFrameParametersValidationErrorMessages
          .labelNameWithoutHeader()));
    });

    test('should return proper error message if `labelName` is provided, '
        'but `headerExists` is false (no header case), despite of not null '
        '`labelIdx`', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final ranges = [
        const Tuple2<int, int>(0, 5),
        const Tuple2<int, int>(6, 8)
      ];
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 3,
        labelName: 'some name',
        columns: ranges,
        headerExists: false,
      );
      expect(actual, equals(DataFrameParametersValidationErrorMessages
          .labelNameWithoutHeader()));
    });
  });

  group('DataFrameParamsValidatorImpl (`headerExists` param)', () {
    test('should return no error message if `headerExists` param is provided',
        () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 10,
        headerExists: true,
      );
      expect(actual, equals(DataFrameParametersValidationErrorMessages
          .noErrorMsg));
    });

    test('should return proper error message if no `headerExists` param '
        'is provided', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final actual = dataFrameParamsValidator.validate(
        labelIdx: null,
        headerExists: null,
      );
      expect(
          actual,
          equals(DataFrameParametersValidationErrorMessages
              .noHeaderExistsParameterProvidedMsg()));
    });
  });

  group('DataFrameParamsValidatorImpl (`predefinedCategories` param)', () {
    test('should return no error message if `predefinedCategories` '
            'param is not provided', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 10,
        predefinedCategories: null,
      );
      expect(actual,
          equals(DataFrameParametersValidationErrorMessages.noErrorMsg));
    });

    test('should return proper error message if `predefinedCategories` param '
        'is provided, but it is empty', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 10,
        predefinedCategories: {},
      );
      expect(actual, equals(DataFrameParametersValidationErrorMessages
          .emptyCategoriesMsg()));
    });

    test('should return proper error message if no `headerExists` param is '
        'provided, but predefined categories exist', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final categories = {
        'category': ['val_1', 'val_2']
      };
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 10,
        headerExists: false,
        predefinedCategories: categories,
      );
      expect(actual, equals(DataFrameParametersValidationErrorMessages
          .noHeaderProvidedMsg(categories)));
    });
  });

  group('DataFrameParamsValidatorImpl (`namesToEncoders` param)', () {
    test('should return no-error message if `namesToEncoders` param is not '
        'provided', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 10,
        namesToEncoders: null,
      );
      expect(actual, equals(DataFrameParametersValidationErrorMessages
          .noErrorMsg));
    });

    test('should return proper error message if `namesToEncoders` is '
        'empty', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 10,
        namesToEncoders: {},
      );
      expect(actual, equals(DataFrameParametersValidationErrorMessages
          .emptyEncodersMsg()));
    });

    test('should return proper error message if no `headerExists` param is '
        'provided, but `namesToEncoders` param exists', () {
      final dataFrameParamsValidator = const DataFrameParamsValidatorImpl();
      final encoders = {'cat1': CategoricalDataEncoderType.ordinal};
      final actual = dataFrameParamsValidator.validate(
        labelIdx: 10,
        headerExists: false,
        namesToEncoders: encoders,
      );
      expect(
          actual,
          equals(DataFrameParametersValidationErrorMessages
              .noHeaderProvidedForColumnEncodersMsg(encoders)));
    });
  });
}
