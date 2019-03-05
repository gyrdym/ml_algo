import 'package:ml_algo/src/data_preprocessing/data_frame/read_mask_creator/read_mask_creator_impl.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';
import 'package:tuple/tuple.dart';

import '../../test_utils/mocks.dart';

void main() {
  final loggerMock = LoggerMock();
  final creator = MLDataReadMaskCreatorImpl(loggerMock);

  group('MLDataReadMaskCreatorImpl', () {
    test('should create read mask, case 1', () {
      final mask = creator.create([const Tuple2<int, int>(0, 7)]);
      expect(mask, equals([true, true, true, true, true, true, true, true]));
    });

    test('should create read mask, case 2', () {
      final mask = creator.create([const Tuple2<int, int>(0, 6)]);
      expect(mask, equals([true, true, true, true, true, true, true]));
    });

    test('should create read mask, case 3', () {
      final mask = creator.create([const Tuple2<int, int>(0, 0)]);
      expect(mask, equals([true]));
    });

    test('should create read mask, case 4', () {
      expect(() => creator.create([]), throwsException);
      verify(loggerMock.severe(MLDataReadMaskCreatorImpl.emptyRangesMsg, any))
          .called(1);
    });

    test('should create read mask, case 5', () {
      final mask = creator
          .create([const Tuple2<int, int>(0, 0), const Tuple2<int, int>(0, 0)]);
      expect(mask, equals([true]));
    });

    test('should create read mask, case 5', () {
      final mask = creator
          .create([const Tuple2<int, int>(0, 0), const Tuple2<int, int>(3, 4)]);
      expect(mask, equals([true, false, false, true, true]));
    });
  });
}
