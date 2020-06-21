class InvalidTrainDataColumnsNumberException implements Exception {
  InvalidTrainDataColumnsNumberException(int expected, int received) :
      message = 'Unexpected columns number in training data, '
          'expected $expected, received ${received}';

  final String message;

  @override
  String toString() => message;
}
