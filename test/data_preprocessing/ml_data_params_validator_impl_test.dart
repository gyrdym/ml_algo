import 'package:ml_algo/src/data_preprocessing/ml_data/validator/error_messages.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/validator/ml_data_params_validator_impl.dart';
import 'package:test/test.dart';
import 'package:tuple/tuple.dart';

void main() {
  group('MLDataParamsValidatorImpl', () {
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

    test('should return no error message if empty row ranges list is provided', () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final actual = mlDataParamsValidator.validate(
        labelIdx: 0,
        rows: [],
        columns: [const Tuple2<int, int>(0, 1)],
        headerExists: true,
      );
      expect(actual, equals(MLDataValidationErrorMessages.noErrorMsg));
    });

    test('should return proper error message if provided row ranges are intersecting', () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final rowRange1 = const Tuple2<int, int>(0, 2);
      final rowRange2 = const Tuple2<int, int>(0, 1);
      final actual = mlDataParamsValidator.validate(
        labelIdx: 0,
        rows: [rowRange1, rowRange2],
        columns: [const Tuple2<int, int>(0, 1)],
        headerExists: true,
      );
      expect(actual, equals(MLDataValidationErrorMessages.intersectingRangesMsg(rowRange1, rowRange2)));
    });

    test('should return proper error message if provided row ranges are intersecting, corner case', () {
      final mlDataParamsValidator = const MLDataParamsValidatorImpl();
      final rowRange1 = const Tuple2<int, int>(0, 2);
      final rowRange2 = const Tuple2<int, int>(2, 3);
      final actual = mlDataParamsValidator.validate(
        labelIdx: 0,
        rows: [rowRange1, rowRange2],
        columns: [const Tuple2<int, int>(0, 1)],
        headerExists: true,
      );
      expect(actual, equals(MLDataValidationErrorMessages.intersectingRangesMsg(rowRange1, rowRange2)));
    });

    test('should return proper error message if at least one of the provided row ranges has its left bondary greater '
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
      expect(actual, equals(MLDataValidationErrorMessages.leftBoundGreaterThanRightMsg(rowRange2)));
    });

    test('should return no error message if valid row ranges list is provided', () {
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
}
