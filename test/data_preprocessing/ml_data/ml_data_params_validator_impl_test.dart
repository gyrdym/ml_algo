import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/validator/error_messages.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/validator/ml_data_params_validator_impl.dart';
import 'package:test/test.dart';
import 'package:tuple/tuple.dart';

void main() {
  group('MLDataParamsValidatorImpl (`rows` param)', () {
    test('should return no error message if no row ranges are provided', () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final actual = mlDataParamsValidator.validate(
        labelIdx: 0,
        rows: null,
        columns: [const Tuple2<int, int>(0, 1)],
        headerExists: true,
      );
      expect(actual, equals(MLDataValidationErrorMessages.noErrorMsg));
    });

    test('should return no error message if empty row ranges list is provided',
        () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final actual = mlDataParamsValidator.validate(
        labelIdx: 0,
        rows: [],
        columns: [const Tuple2<int, int>(0, 1)],
        headerExists: true,
      );
      expect(actual, equals(MLDataValidationErrorMessages.noErrorMsg));
    });

    test(
        'should return proper error message if provided row ranges are intersecting',
        () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final rowRange1 = const Tuple2<int, int>(0, 2);
      final rowRange2 = const Tuple2<int, int>(0, 1);
      final actual = mlDataParamsValidator.validate(
        labelIdx: 0,
        rows: [rowRange1, rowRange2],
        columns: [const Tuple2<int, int>(0, 1)],
        headerExists: true,
      );
      expect(
          actual,
          equals(MLDataValidationErrorMessages.intersectingRangesMsg(
              rowRange1, rowRange2)));
    });

    test(
        'should return proper error message if provided row ranges are intersecting, corner case',
        () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final rowRange1 = const Tuple2<int, int>(0, 2);
      final rowRange2 = const Tuple2<int, int>(2, 3);
      final actual = mlDataParamsValidator.validate(
        labelIdx: 0,
        rows: [rowRange1, rowRange2],
        columns: [const Tuple2<int, int>(0, 1)],
        headerExists: true,
      );
      expect(
          actual,
          equals(MLDataValidationErrorMessages.intersectingRangesMsg(
              rowRange1, rowRange2)));
    });

    test(
        'should return proper error message if at least one of the provided row ranges has its left bondary greater '
        'than the right', () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final rowRange1 = const Tuple2<int, int>(0, 2);
      final rowRange2 = const Tuple2<int, int>(5, 3);
      final actual = mlDataParamsValidator.validate(
        labelIdx: 0,
        rows: [rowRange1, rowRange2],
        columns: [const Tuple2<int, int>(0, 1)],
        headerExists: true,
      );
      expect(
          actual,
          equals(MLDataValidationErrorMessages.leftBoundGreaterThanRightMsg(
              rowRange2)));
    });

    test('should return no error message if valid row ranges list is provided',
        () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final rowRange1 = const Tuple2<int, int>(0, 2);
      final rowRange2 = const Tuple2<int, int>(5, 6);
      final rowRange3 = const Tuple2<int, int>(7, 10);
      final actual = mlDataParamsValidator.validate(
        labelIdx: 0,
        rows: [rowRange1, rowRange2, rowRange3],
        columns: [const Tuple2<int, int>(0, 1)],
        headerExists: true,
      );
      expect(actual, equals(MLDataValidationErrorMessages.noErrorMsg));
    });
  });

  group('MLDataParamsValidatorImpl (`columns` param)', () {
    test('should return no error message if no row ranges are provided', () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final actual = mlDataParamsValidator.validate(
        labelIdx: 0,
        rows: [const Tuple2<int, int>(0, 1)],
        columns: null,
        headerExists: true,
      );
      expect(actual, equals(MLDataValidationErrorMessages.noErrorMsg));
    });

    test('should return no error message if empty row ranges list is provided',
        () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final actual = mlDataParamsValidator.validate(
        labelIdx: 0,
        rows: [const Tuple2<int, int>(0, 1)],
        columns: [],
        headerExists: true,
      );
      expect(actual, equals(MLDataValidationErrorMessages.noErrorMsg));
    });

    test(
        'should return proper error message if provided row ranges are intersecting',
        () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final colRange1 = const Tuple2<int, int>(0, 2);
      final colRange2 = const Tuple2<int, int>(0, 1);
      final actual = mlDataParamsValidator.validate(
        labelIdx: 0,
        rows: [const Tuple2<int, int>(0, 1)],
        columns: [colRange1, colRange2],
        headerExists: true,
      );
      expect(
          actual,
          equals(MLDataValidationErrorMessages.intersectingRangesMsg(
              colRange1, colRange2)));
    });

    test(
        'should return proper error message if provided row ranges are intersecting, corner case',
        () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final colRange1 = const Tuple2<int, int>(0, 2);
      final colRange2 = const Tuple2<int, int>(2, 3);
      final actual = mlDataParamsValidator.validate(
        labelIdx: 0,
        rows: [const Tuple2<int, int>(0, 1)],
        columns: [colRange1, colRange2],
        headerExists: true,
      );
      expect(
          actual,
          equals(MLDataValidationErrorMessages.intersectingRangesMsg(
              colRange1, colRange2)));
    });

    test(
        'should return proper error message if at least one of the provided row ranges has its left bondary greater '
        'than the right', () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final colRange1 = const Tuple2<int, int>(0, 2);
      final colRange2 = const Tuple2<int, int>(5, 3);
      final actual = mlDataParamsValidator.validate(
        labelIdx: 0,
        rows: [const Tuple2<int, int>(0, 1)],
        columns: [colRange1, colRange2],
        headerExists: true,
      );
      expect(
          actual,
          equals(MLDataValidationErrorMessages.leftBoundGreaterThanRightMsg(
              colRange2)));
    });

    test('should return no error message if valid row ranges list is provided',
        () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final colRange1 = const Tuple2<int, int>(0, 2);
      final colRange2 = const Tuple2<int, int>(5, 6);
      final colRange3 = const Tuple2<int, int>(7, 10);
      final actual = mlDataParamsValidator.validate(
        labelIdx: 0,
        rows: [const Tuple2<int, int>(0, 1)],
        columns: [colRange1, colRange2, colRange3],
        headerExists: true,
      );
      expect(actual, equals(MLDataValidationErrorMessages.noErrorMsg));
    });
  });

  group('MLDataParamsValidatorImpl (`labelIdx` param)', () {
    test('should return no error message if label index is provided', () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final actual = mlDataParamsValidator.validate(
        labelIdx: 10,
        headerExists: true,
      );
      expect(actual, equals(MLDataValidationErrorMessages.noErrorMsg));
    });

    test('should return proper error message if no label index is provided',
        () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final actual = mlDataParamsValidator.validate(
        labelIdx: null,
        headerExists: true,
      );
      expect(actual, equals(MLDataValidationErrorMessages.noLabelIndexMsg()));
    });

    test(
        'should return proper error message if provided column ranges list does not contain label index',
        () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final labelIdx = 10;
      final ranges = [
        const Tuple2<int, int>(0, 5),
        const Tuple2<int, int>(6, 8)
      ];
      final actual = mlDataParamsValidator.validate(
        labelIdx: labelIdx,
        columns: ranges,
        headerExists: true,
      );
      expect(
          actual,
          equals(MLDataValidationErrorMessages.labelIsNotInRangesMsg(
              labelIdx, ranges)));
    });
  });

  group('MLDataParamsValidatorImpl (`headerExists` param)', () {
    test('should return no error message if `headerExists` param is provided',
        () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final actual = mlDataParamsValidator.validate(
        labelIdx: 10,
        headerExists: true,
      );
      expect(actual, equals(MLDataValidationErrorMessages.noErrorMsg));
    });

    test(
        'should return proper error message if no `headerExists` param is provided',
        () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final actual = mlDataParamsValidator.validate(
        labelIdx: null,
        headerExists: null,
      );
      expect(
          actual,
          equals(MLDataValidationErrorMessages
              .noHeaderExistsParameterProvidedMsg()));
    });
  });

  group('MLDataParamsValidatorImpl (`predefinedCategories` param)', () {
    test(
        'should return no error message if `predefinedCategories` param is not provided',
        () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final actual = mlDataParamsValidator.validate(
        labelIdx: 10,
        predefinedCategories: null,
      );
      expect(actual, equals(MLDataValidationErrorMessages.noErrorMsg));
    });

    test(
        'should return proper error message if `predefinedCategories` param is provided, but it is empty',
        () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final actual = mlDataParamsValidator.validate(
        labelIdx: 10,
        predefinedCategories: {},
      );
      expect(
          actual, equals(MLDataValidationErrorMessages.emptyCategoriesMsg()));
    });

    test(
        'should return proper error message if no `headerExists` param is provided, but predefined categories '
        'exist', () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final categories = {
        'category': ['val_1', 'val_2']
      };
      final actual = mlDataParamsValidator.validate(
        labelIdx: 10,
        headerExists: false,
        predefinedCategories: categories,
      );
      expect(
          actual,
          equals(
              MLDataValidationErrorMessages.noHeaderProvidedMsg(categories)));
    });
  });

  group('MLDataParamsValidatorImpl (`namesToEncoders` param)', () {
    test(
        'should return no-error message if `namesToEncoders` param is not provided',
        () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final actual = mlDataParamsValidator.validate(
        labelIdx: 10,
        namesToEncoders: null,
      );
      expect(actual, equals(MLDataValidationErrorMessages.noErrorMsg));
    });

    test('should return proper error message if `namesToEncoders` is empty',
        () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final actual = mlDataParamsValidator.validate(
        labelIdx: 10,
        namesToEncoders: {},
      );
      expect(actual, equals(MLDataValidationErrorMessages.emptyEncodersMsg()));
    });

    test(
        'should return proper error message if no `headerExists` param is provided, but `namesToEncoders` param '
        'exists', () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final encoders = {'cat1': CategoricalDataEncoderType.ordinal};
      final actual = mlDataParamsValidator.validate(
        labelIdx: 10,
        headerExists: false,
        namesToEncoders: encoders,
      );
      expect(
          actual,
          equals(MLDataValidationErrorMessages
              .noHeaderProvidedForColumnEncodersMsg(encoders)));
    });
  });
}
