class InvalidTrainDataColumnsNumberException implements Exception {
  InvalidTrainDataColumnsNumberException(int expected, int received) :
      message = 'Unexpected columns number in train data, '
          'expected $expected, received $received';

  final String message;

  @override
  String toString() => message;
}
